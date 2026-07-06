"""SceneEraser v13 — 영상 동적 객체(행인) 자동 제거 파이프라인.

고정 카메라 영상에서 카메라 앞을 지나가는 행인(REMOVE)은 지우고,
그림을 보는 관람객처럼 거의 움직이지 않는 정적 인물(KEEP)은 보존한다.

파이프라인 단계
    1. preprocess_video   해상도(최대 1920x1080)·길이(최대 30초) 제한
    2. extract_frames     JPEG 프레임 추출
    3. detect_track       YOLOv8n + ByteTrack 사람 검출/추적
    4. merge_fragments    끊긴 track을 같은 사람으로 병합(occlusion·ID swap 보정)
    5. score_select       이동량·위치 기반 점수화로 REMOVE 대상 자동 선정
    6. generate_masks     SAM2(GPU) 또는 bbox 마스크 생성
    7. refine_masks       morphology + 시간 보팅으로 마스크 안정화
    8. build_temporal_plate  프레임별 중앙값으로 정적 배경 plate 추정
    9. inpaint_plate      plate의 잔여 구멍을 cv2.inpaint(TELEA)로 채움
   10. build_static_plate 정적 배경 픽셀 판별(합성 1순위 fill 소스)
   11. expand_shadow_masks  HSV 기반 그림자 검출로 마스크 확장
   12. composite_with_mode  다른 프레임/plate에서 픽셀을 가져와 최종 합성
   13. mux_audio         원본 오디오 재결합

진입점: run_pipeline()  —  app.py의 Gradio UI가 호출한다.
로직은 v12(make_v12.py)와 동일하며, 노트북 셀 구조만 표준 모듈로 정리했다.
"""

import bisect
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PERSON_CLASS = 0
CONF_THRES   = 0.25
TRACKER_CFG  = "bytetrack.yaml"
MAX_W, MAX_H, MAX_SEC = 1920, 1080, 30


def _odd(k):
    k = max(int(k), 1)
    return k if k % 2 == 1 else k + 1

def _sorted_frames(d):
    return sorted(Path(d).glob("*.jpg"))


# ── 1. 영상 전처리 ──────────────────────────────────────────────────────────────
def preprocess_video(src, dst):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("영상 열기 실패")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(MAX_W/w, MAX_H/h, 1.0)
    tw, th = int(w*scale)//2*2, int(h*scale)//2*2
    max_frames = int(fps * MAX_SEC)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (tw, th))
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        if scale < 1.0:
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
    cap.release()
    writer.release()
    return {"fps": fps, "w": tw, "h": th, "n": count}


# ── 2. 프레임 추출 ──────────────────────────────────────────────────────────────
def extract_frames(video, frames_dir):
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(str(Path(frames_dir) / f"{i:06d}.jpg"), frame)
        i += 1
    cap.release()
    return i


# ── 3. YOLO + ByteTrack ─────────────────────────────────────────────────────────
def detect_track(video, model_name="yolov8n.pt"):
    yolo = YOLO(model_name)
    cap  = cv2.VideoCapture(str(video))
    rows, fi = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = yolo.track(source=frame, persist=True, tracker=TRACKER_CFG,
                         conf=CONF_THRES, classes=[PERSON_CLASS], verbose=False)
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            for box, tid, conf in zip(
                res[0].boxes.xyxy.cpu().numpy(),
                res[0].boxes.id.cpu().numpy().astype(int),
                res[0].boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                bw, bh = x2-x1, y2-y1
                rows.append({"frame_idx": fi, "track_id": int(tid), "conf": float(conf),
                             "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "cx": (x1+x2)/2, "cy": (y1+y2)/2, "w": bw, "h": bh})
        fi += 1
    cap.release()
    return pd.DataFrame(rows)


# ── 4. merge_fragments (track → group) ─────────────────────────────────────────
def _track_summary(df):
    rows = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("frame_idx")
        f, l = g.iloc[0], g.iloc[-1]
        mw, mh = float(g["w"].mean()), float(g["h"].mean())
        rows.append({
            "track_id": int(tid), "frame_count": len(g),
            "first_frame": int(f["frame_idx"]), "last_frame": int(l["frame_idx"]),
            "first_cx": float(f["cx"]), "first_cy": float(f["cy"]),
            "last_cx":  float(l["cx"]), "last_cy":  float(l["cy"]),
            "mean_w": mw, "mean_h": mh, "mean_diag": float(np.sqrt(mw**2+mh**2)),
        })
    return pd.DataFrame(rows).sort_values(["first_frame","track_id"]).reset_index(drop=True)


def merge_fragments(df, min_frames=4, max_gap=24, max_ndist=0.75, max_sratio=0.45):
    df = df[df["track_id"] >= 0].copy()
    vc = df["track_id"].value_counts()
    df = df[df["track_id"].isin(vc[vc >= min_frames].index)].copy()
    if df.empty:
        df["group_id"] = pd.Series(dtype=int)
        return df
    summ = _track_summary(df)
    rows = summ.to_dict("records")
    ffs  = [r["first_frame"] for r in rows]
    candidates = []
    for i, a in enumerate(rows):
        lo = bisect.bisect_right(ffs, a["last_frame"])
        hi = bisect.bisect_right(ffs, a["last_frame"] + max_gap)
        for b in rows[lo:hi]:
            dx, dy = b["first_cx"]-a["last_cx"], b["first_cy"]-a["last_cy"]
            nd = np.sqrt(dx*dx+dy*dy) / max((a["mean_diag"]+b["mean_diag"])*0.5, 1.0)
            sw = abs(a["mean_w"]-b["mean_w"]) / max((a["mean_w"]+b["mean_w"])*0.5, 1.0)
            sh = abs(a["mean_h"]-b["mean_h"]) / max((a["mean_h"]+b["mean_h"])*0.5, 1.0)
            if nd <= max_ndist and max(sw, sh) <= max_sratio:
                candidates.append({"src": a["track_id"], "dst": b["track_id"],
                                   "gap": b["first_frame"]-a["last_frame"], "nd": float(nd)})
    candidates.sort(key=lambda x: (x["nd"], x["gap"]))
    succ, pred, valid = {}, {}, set(summ["track_id"])
    for c in candidates:
        s, d = c["src"], c["dst"]
        if s not in valid or d not in valid or s in succ or d in pred: continue
        succ[s] = d; pred[d] = s
    ff_map = dict(zip(summ["track_id"], summ["first_frame"]))
    gmap, gid = {}, 0
    for start in sorted([t for t in valid if t not in pred], key=lambda t: ff_map[t]):
        chain, cur = [start], start
        while cur in succ:
            cur = succ[cur]
            if cur in chain: break
            chain.append(cur)
        for t in chain: gmap[int(t)] = gid
        gid += 1
    for t in sorted(valid):
        if int(t) not in gmap: gmap[int(t)] = gid; gid += 1
    df = df.copy()
    df["group_id"] = df["track_id"].map(gmap)
    # ── 진단 출력 (v10_J) ─────────────────────────────────────────────────────
    print("\n" + "="*78)
    print(f"[merge_fragments 진단]  track {df['track_id'].nunique()}개 "
          f"→ group {df['group_id'].nunique()}개")
    for gid_, g in df.groupby("group_id"):
        tids = sorted(g["track_id"].unique().tolist())
        print(f"  group {gid_}: track {tids}  "
              f"(frame {int(g['frame_idx'].min())}~{int(g['frame_idx'].max())})")
    print("="*78)
    return df


# ── 5. score_select ─────────────────────────────────────────────────────────────
def _norm(s):
    mn, mx = s.min(), s.max()
    return (s-mn)/(mx-mn) if mx > mn else s*0


def score_select(grouped, top_k, fw, fh):
    sd = []; cr = 0.4
    ex, ey = fw*0.15, fh*0.15
    cx_lo, cx_hi = fw*(0.5-cr/2), fw*(0.5+cr/2)
    cy_lo, cy_hi = fh*(0.5-cr/2), fh*(0.5+cr/2)
    for gid, g in grouped.groupby("group_id"):
        g = g.sort_values("frame_idx")
        cx_f, cy_f = g.iloc[0]["cx"], g.iloc[0]["cy"]
        cx_l, cy_l = g.iloc[-1]["cx"], g.iloc[-1]["cy"]
        nn  = np.sqrt((cx_l-cx_f)**2 + (cy_l-cy_f)**2)
        xr  = (g["cx"].max() - g["cx"].min()) / fw
        edge = 1.0 if (cx_f<ex or cx_f>fw-ex or cx_l<ex or cx_l>fw-ex) else 0.0
        cov  = g["frame_idx"].nunique() / max(grouped["frame_idx"].nunique(), 1)
        cdw  = ((g["cx"]>cx_lo)&(g["cx"]<cx_hi)&(g["cy"]>cy_lo)&(g["cy"]<cy_hi)).mean()
        sd.append({"gid": int(gid), "nn": nn, "xr": xr, "edge": edge,
                   "cov": cov, "cdw": cdw, "nf": g["track_id"].nunique(),
                   "f_start": int(g["frame_idx"].min()), "f_end": int(g["frame_idx"].max()),
                   "cx_min": float(g["cx"].min()), "cx_max": float(g["cx"].max())})
    sd = pd.DataFrame(sd)
    if sd.empty: return []
    sd["score"] = (0.35*_norm(sd["nn"]) + 0.25*_norm(sd["xr"]) + 0.15*sd["edge"]
                   + 0.10*(1-_norm(sd["cov"])) + 0.10*(1-_norm(sd["cdw"]))
                   + 0.05*(1-_norm(sd["nf"])))
    # ── 진단 출력 (v10_J) ─────────────────────────────────────────────────────
    diag = sd.sort_values("score", ascending=False)
    print("\n" + "="*78)
    print(f"[score_select 진단]  화면: {fw}x{fh}  그룹수: {len(sd)}  top_k: {top_k}")
    print(f"{'gid':>4} {'score':>7} | {'nn':>7} {'xr':>6} {'edge':>5} {'cov':>6} "
          f"{'cdw':>6} {'nf':>3} | {'frame':>11} {'cx범위':>13}")
    print("-"*78)
    for _, r in diag.iterrows():
        print(f"{int(r['gid']):>4} {r['score']:>7.3f} | "
              f"{r['nn']:>7.1f} {r['xr']:>6.3f} {r['edge']:>5.1f} "
              f"{r['cov']:>6.3f} {r['cdw']:>6.3f} {int(r['nf']):>3} | "
              f"{int(r['f_start']):>4}~{int(r['f_end']):<4} "
              f"{int(r['cx_min']):>5}~{int(r['cx_max']):<5}")
    selected = diag.head(top_k)["gid"].astype(int).tolist()
    print(f"→ 선택된 제거 대상: {selected}")
    print("="*78 + "\n")
    return selected


# ── 6. generate_masks (SAM2 또는 bbox) ─────────────────────────────────────────
def _sam2_available():
    try:
        import sam2  # noqa
        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_masks(frames_dir, remove_df, masks_dir, mode="auto", sam2_ckpt=None):
    masks_dir = Path(masks_dir); masks_dir.mkdir(parents=True, exist_ok=True)
    fps = _sorted_frames(frames_dir)
    if not fps: return
    h, w = cv2.imread(str(fps[0])).shape[:2]
    if mode == "auto": mode = "sam2" if _sam2_available() else "bbox"
    if mode == "bbox" or remove_df.empty:
        for fp in fps:
            fi = int(fp.stem); m = np.zeros((h, w), np.uint8)
            for _, r in remove_df[remove_df["frame_idx"] == fi].iterrows():
                cv2.rectangle(m, (int(r["x1"]),int(r["y1"])), (int(r["x2"]),int(r["y2"])), 255, -1)
            m = cv2.dilate(m, np.ones((9,9), np.uint8), iterations=1)
            cv2.imwrite(str(masks_dir / f"{fi:06d}.png"), m)
        return
    from sam2.build_sam import build_sam2_video_predictor
    ckpt = sam2_ckpt or "/content/sam2_small.pt"
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", ckpt, device="cuda")
    df2 = remove_df.sort_values(["track_id","frame_idx","conf"], ascending=[True,True,False])
    prompts = [g.iloc[0] for _, g in df2.groupby("track_id")]
    seg_per_frame = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir))
        for obj_id, row in enumerate(prompts, 1):
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=int(row["frame_idx"]), obj_id=obj_id,
                box=np.array([row["x1"],row["y1"],row["x2"],row["y2"]], np.float32))
        for fi, _ids, logits in predictor.propagate_in_video(state):
            if isinstance(logits, torch.Tensor): logits = logits.float().cpu().numpy()
            union = np.zeros((h, w), bool)
            for k in range(logits.shape[0]): union |= (logits[k, 0] > 0.0)
            seg_per_frame[int(fi)] = union.astype(np.uint8) * 255
    kernel8 = np.ones((9,9), np.uint8)
    for fp in fps:
        fi = int(fp.stem)
        m = seg_per_frame.get(fi, np.zeros((h, w), np.uint8))
        m = cv2.dilate(m, kernel8, iterations=1)
        cv2.imwrite(str(masks_dir / f"{fi:06d}.png"), m)


# ── 7. refine_masks (v10_J: stab_mode + tight) ─────────────────────────────────
def refine_masks(frames_dir, masks_dir, refined_dir, stab_mode="legacy", tight=False):
    """
    stab_mode:
        "legacy"        — 기존. 인접 3프레임 중 2개 이상에 마스크가 있어야 통과.
        "bidirectional" — 양방향. 빠진 프레임 복구 + 고립 프레임 제거.
    tight: True면 dilate 축소 (9→5, bottom 25→10). SAM2 정밀 마스크에서 배경 침범 감소.
    """
    refined_dir = Path(refined_dir); refined_dir.mkdir(parents=True, exist_ok=True)
    mps  = sorted(Path(masks_dir).glob("*.png"))
    raws = [cv2.imread(str(p), 0) for p in mps]
    dil_size  = 5 if tight else 9
    bottom_px = 10 if tight else 25
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(7), _odd(7)))
    k_dil   = np.ones((_odd(dil_size), _odd(dil_size)), np.uint8)
    proc = []
    for m in raws:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        m = cv2.dilate(m, k_dil, iterations=1)
        sh = np.zeros_like(m); sh[bottom_px:, :] = m[:-bottom_px, :]
        proc.append(np.maximum(m, sh))
    n = len(proc); stab = []
    if stab_mode == "legacy":
        for i in range(n):
            votes = sum(1 for j in range(max(0,i-1), min(n,i+2)) if proc[j].any())
            stab.append(proc[i] if votes >= 2 else np.zeros_like(proc[i]))
    else:  # bidirectional
        for i in range(n):
            cur  = proc[i].any()
            prev = proc[i-1].any() if i > 0     else False
            nxt  = proc[i+1].any() if i < n-1   else False
            if cur:
                stab.append(np.zeros_like(proc[i]) if (not prev and not nxt) else proc[i])
            else:
                stab.append(cv2.bitwise_or(proc[i-1], proc[i+1]) if (prev and nxt) else proc[i])
    for i, m in enumerate(stab): cv2.imwrite(str(refined_dir / f"{i:06d}.png"), m)
    return stab


# ── 8. build_temporal_plate ─────────────────────────────────────────────────────
def build_temporal_plate(frames, masks):
    h, w = frames[0].shape[:2]
    plate = np.zeros((h, w, 3), np.uint8); residual = np.zeros((h, w), bool)
    CHUNK = 64; masks_bool = [(m > 0) for m in masks]
    for y0 in range(0, h, CHUNK):
        y1 = min(y0+CHUNK, h)
        chunk = np.stack([f[y0:y1] for f in frames]).astype(np.float32)
        cmask = np.stack([m[y0:y1] for m in masks_bool])
        chunk = np.where(cmask[..., None], np.nan, chunk)
        with np.errstate(all="ignore"): med = np.nanmedian(chunk, axis=0)
        nan_all = np.isnan(med).any(axis=-1)
        residual[y0:y1] = nan_all; med[np.isnan(med)] = 0
        plate[y0:y1] = np.clip(med, 0, 255).astype(np.uint8)
    return plate, residual


# ── 9. inpaint_plate ────────────────────────────────────────────────────────────
def inpaint_plate(plate, residual):
    if not residual.any(): return plate
    return cv2.inpaint(plate, residual.astype(np.uint8)*255, 5, cv2.INPAINT_TELEA)


# ── 10. 정적 배경 픽셀 판별 (v9) ────────────────────────────────────────────────
def build_static_plate(frames, masks, diff_thresh=10):
    f0 = frames[0].astype(np.float32); fn = frames[-1].astype(np.float32)
    diff = np.abs(f0 - fn).max(axis=-1)
    clean_0 = ~(masks[0]  > 0); clean_n = ~(masks[-1] > 0)
    static_mask  = (diff < diff_thresh) & clean_0 & clean_n
    static_color = ((f0 + fn) / 2).clip(0, 255).astype(np.uint8)
    return static_mask, static_color


# ── 11. 전역 LAB affine plate 보정 (v12, Option C) ──────────────────────────────
def _global_lab_correction(plate, frame, mask_u8):
    """
    mask 밖 전체 픽셀에서 plate→frame LAB affine(scale+offset) 추정.
    전역 조명 변화(구름, 밝기 drift, 화이트밸런스 drift)를 보정.
    기존 BGR scale-only 보정(_local_plate_correction) 대체.
    """
    outside = ~(mask_u8 > 0)
    if outside.sum() < 200:
        return plate
    plate_lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    corrected = plate_lab.copy()
    for c in range(3):
        px = plate_lab[:, :, c][outside]
        fx = frame_lab[:, :, c][outside]
        if px.std() > 1.0:
            a, b = np.polyfit(px, fx, 1)
            a = float(np.clip(a, 0.7, 1.4))
            b = float(np.clip(b, -30.0, 30.0))
            corrected[:, :, c] = np.clip(plate_lab[:, :, c] * a + b, 0, 255)
    return cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ── 12. 마스크 확장: HSV 그림자 검출 (v9) + cap 옵션 (v10_J) ───────────────────
def expand_shadow_masks(frames, refined, plate,
                        search_px=60, shadow_v_thresh=20, shadow_hue_thresh=25,
                        static_diff_thresh=10, cap=False, cap_ratio=0.15):
    """
    HSV 기반 그림자 검출 + connected components 필터링 (v9).
    cap: True면 한 프레임 추가량이 마스크 면적의 cap_ratio 배 넘으면 확장 스킵 (v10_J).
    """
    out = []
    k_close   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(11), _odd(11)))
    bridge_sz = _odd(search_px // 3)
    bridge_k  = np.ones((bridge_sz, bridge_sz), np.uint8)
    static_mask, static_color = build_static_plate(frames, refined, diff_thresh=static_diff_thresh)
    ref_color = plate.copy(); ref_color[static_mask] = static_color[static_mask]
    ref_hsv   = cv2.cvtColor(ref_color, cv2.COLOR_BGR2HSV).astype(np.int16)
    SHADOW_V_STATIC   = max(shadow_v_thresh // 2, 8)
    SHADOW_HUE_STATIC = shadow_hue_thresh + 5

    for frame, m in zip(frames, refined):
        if not m.any(): out.append(m); continue
        f_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        v_drop = ref_hsv[:, :, 2] - f_hsv[:, :, 2]
        h_diff = np.abs(ref_hsv[:, :, 0] - f_hsv[:, :, 0])
        h_diff = np.minimum(h_diff, 180 - h_diff)
        is_shadow = (
            (static_mask  & (v_drop > SHADOW_V_STATIC)  & (h_diff < SHADOW_HUE_STATIC)) |
            (~static_mask & (v_drop > shadow_v_thresh)   & (h_diff < shadow_hue_thresh))
        )
        combined = np.maximum(m, is_shadow.astype(np.uint8) * 255)
        seeded      = np.maximum(cv2.dilate(m, bridge_k, iterations=1), combined)
        _, labels   = cv2.connectedComponents(seeded)
        orig_labels = np.unique(labels[m > 0]); orig_labels = orig_labels[orig_labels > 0]
        if orig_labels.size > 0:
            in_group = np.isin(labels, orig_labels)
            combined = (in_group & (combined > 0)).astype(np.uint8) * 255
            combined = np.maximum(combined, m)
        else:
            combined = m.copy()
        if cap:
            m_area = int((m > 0).sum())
            added  = int((combined > 0).sum()) - m_area
            if m_area > 0 and added > m_area * cap_ratio:
                out.append(m); continue
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
        out.append(combined)
    return out


# ── 13. KEEP 객체 bbox 마스크 (v9) ──────────────────────────────────────────────
def build_keep_masks(grouped, remove_ids, n_frames, h, w):
    keep_df = grouped[~grouped["group_id"].isin(remove_ids)]
    masks   = [np.zeros((h, w), bool) for _ in range(n_frames)]
    for _, row in keep_df.iterrows():
        fi = int(row["frame_idx"])
        if fi >= n_frames: continue
        x1 = max(0, int(row["x1"])); y1 = max(0, int(row["y1"]))
        x2 = min(w, int(row["x2"])); y2 = min(h, int(row["y2"]))
        masks[fi][y1:y2, x1:x2] = True
    return masks


# ── 14. 합성 (v9 정교한 로직 + v10_J mode B) ───────────────────────────────────
def composite_with_mode(frames, masks, clean_plate, fps, out, w, h,
                        mode="A", max_search=5,
                        shadow_v_thresh=20, shadow_hue_thresh=25,
                        static_diff_thresh=10, keep_masks=None,
                        residual_search_px=120, feather_sigma=6,
                        gap_feather_px=0,
                        lum_match=True, lum_thresh=10.0):
    """
    A: 다른 프레임 우선 (시간 무한대), plate fallback
    B: plate-only. mask 영역 전부 clean_plate로 교체
    C: 5프레임 안에서만 시도, 못 채우면 plate
    v9 핵심: 정적 배경 1순위 fill · keep 객체 우회 · donor 그림자 필터 · feathering
    v12 신기능:
      lum_match: 조명 유사 프레임(LAB L 평균 차이 < lum_thresh)에서만 차용 (Option B)
      plate 보정: _global_lab_correction (전역 LAB affine, Option C)
    """
    n = len(frames)
    mask_bool = [m.astype(bool) for m in masks]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # mode B: 가장 단순 — 즉시 처리
    if mode == "B":
        for i in range(n):
            result = frames[i].copy()
            if mask_bool[i].any():
                result[mask_bool[i]] = clean_plate[mask_bool[i]]
            writer.write(result)
        writer.release(); return

    static_mask, static_color = build_static_plate(frames, masks, diff_thresh=static_diff_thresh)
    static_px = int(static_mask.sum())
    print(f"  정적 배경 픽셀: {static_px:,} / {w*h:,} ({static_px/(w*h)*100:.1f}%)")

    ref_color = clean_plate.copy(); ref_color[static_mask] = static_color[static_mask]
    ref_hsv   = cv2.cvtColor(ref_color, cv2.COLOR_BGR2HSV).astype(np.int16)
    SHADOW_V_STATIC   = max(shadow_v_thresh // 2, 8)
    SHADOW_HUE_STATIC = shadow_hue_thresh + 5
    shadow_maps = []
    for f in frames:
        f_hsv  = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.int16)
        v_drop = ref_hsv[:, :, 2] - f_hsv[:, :, 2]
        h_diff = np.abs(ref_hsv[:, :, 0] - f_hsv[:, :, 0])
        h_diff = np.minimum(h_diff, 180 - h_diff)
        shadow_maps.append(
            (static_mask  & (v_drop > SHADOW_V_STATIC)  & (h_diff < SHADOW_HUE_STATIC)) |
            (~static_mask & (v_drop > shadow_v_thresh)   & (h_diff < shadow_hue_thresh))
        )
    plate_hsv       = cv2.cvtColor(clean_plate, cv2.COLOR_BGR2HSV).astype(np.int16)
    bridge_sz       = _odd(residual_search_px // 3)
    residual_bridge = np.ones((bridge_sz, bridge_sz), np.uint8)
    gap_kernel = (np.ones((_odd(gap_feather_px*2), _odd(gap_feather_px*2)), np.uint8)
                  if gap_feather_px > 0 else None)

    # Option B: 프레임별 평균 밝기 사전 계산 (LAB L채널, mask 합집합 밖에서)
    if lum_match:
        _mask_union = np.zeros((h, w), bool)
        for m in mask_bool: _mask_union |= m
        _outside = ~_mask_union
        mean_lums = []
        for f in frames:
            _lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
            _px  = _lab[:, :, 0][_outside].astype(np.float32)
            mean_lums.append(float(_px.mean()) if _px.size > 0 else 128.0)
    else:
        mean_lums = None

    for i in range(n):
        if not mask_bool[i].any(): writer.write(frames[i]); continue
        # Option C: 전역 LAB affine으로 plate를 현재 프레임 조명에 맞게 보정
        local_plate = _global_lab_correction(clean_plate, frames[i], masks[i])
        fill = local_plate.copy(); unfilled = mask_bool[i].copy()
        static_available = static_mask.copy()
        if keep_masks is not None: static_available = static_available & ~keep_masks[i]
        static_here = unfilled & static_available
        if static_here.any(): fill[static_here] = static_color[static_here]; unfilled[static_here] = False
        search_range = range(1, max_search+1) if mode == "C" else range(1, n)
        for d in search_range:
            if not unfilled.any(): break
            for j in (i-d, i+d):
                if 0 <= j < n and unfilled.any():
                    # Option B: 조명 유사도 체크 (LAB L평균 차이 > lum_thresh면 스킵)
                    if mean_lums is not None and abs(mean_lums[i] - mean_lums[j]) > lum_thresh:
                        continue
                    copyable = unfilled & ~mask_bool[j] & ~shadow_maps[j]
                    if copyable.any(): fill[copyable] = frames[j][copyable]; unfilled[copyable] = False
        result = frames[i].copy(); result[mask_bool[i]] = fill[mask_bool[i]]
        shadow_residual = np.zeros(mask_bool[i].shape, bool)
        r_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.int16)
        vd = plate_hsv[:,:,2] - r_hsv[:,:,2]; hd = np.abs(plate_hsv[:,:,0] - r_hsv[:,:,0])
        hd = np.minimum(hd, 180-hd)
        shadow_cand = (~mask_bool[i] & (vd > shadow_v_thresh) & (hd < shadow_hue_thresh)).astype(np.uint8)
        if shadow_cand.any():
            seeded_r  = np.maximum(cv2.dilate(masks[i], residual_bridge, iterations=1)//255, shadow_cand)
            _, lbls_r = cv2.connectedComponents(seeded_r)
            orig_lr   = np.unique(lbls_r[mask_bool[i]]); orig_lr = orig_lr[orig_lr > 0]
            if orig_lr.size > 0:
                shadow_residual = np.isin(lbls_r, orig_lr) & (shadow_cand > 0)
                result[shadow_residual] = local_plate[shadow_residual]
        full_replaced = mask_bool[i] | shadow_residual
        if feather_sigma > 0 and full_replaced.any():
            base_u8 = full_replaced.astype(np.uint8) * 255
            if gap_kernel is not None: base_u8 = cv2.dilate(base_u8, gap_kernel, iterations=1)
            soft = cv2.GaussianBlur(base_u8.astype(np.float32), (0,0), sigmaX=float(feather_sigma)) / 255.0
            soft[full_replaced] = 1.0
            result = (result.astype(np.float32)*soft[...,None] +
                      frames[i].astype(np.float32)*(1.0-soft[...,None])).clip(0,255).astype(np.uint8)
        writer.write(result)
    writer.release()


# ── 15. 원본 오디오 합성 (v10_J) ────────────────────────────────────────────────
def mux_audio(src_video, silent_video, out_video):
    """원본 오디오를 무음 결과 영상에 입힘. ffmpeg 없으면 무음 복사."""
    if shutil.which("ffmpeg") is None:
        shutil.copy(silent_video, out_video); return False
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(src_video)],
        capture_output=True, text=True)
    if not probe.stdout.strip():
        shutil.copy(silent_video, out_video); return False
    cmd = ["ffmpeg", "-y", "-i", str(silent_video), "-i", str(src_video),
           "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest",
           str(out_video)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not Path(out_video).exists():
        shutil.copy(silent_video, out_video); return False
    return True


# ── 16. preview 시각화 (v10_J: 최다 등장 프레임 선택 + IoU 박스 병합) ──────────
def build_preview(frames_dir, grouped, remove_ids):
    frame_files = _sorted_frames(frames_dir)
    if not frame_files: return None
    per_frame = grouped.groupby("frame_idx")["group_id"].nunique()
    bg_idx = int(per_frame.idxmax()) if not per_frame.empty else 0
    preview = cv2.imread(str(Path(frames_dir) / f"{bg_idx:06d}.jpg"))
    if preview is None: preview = cv2.imread(str(frame_files[0]))
    if preview is None: return None
    fr = grouped[grouped["frame_idx"] == bg_idx]
    boxes = []
    for gid, g in fr.groupby("group_id"):
        row = g.iloc[0]
        boxes.append([int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]),
                      int(gid) in remove_ids])
    def iou(a, b):
        ix1,iy1 = max(a[0],b[0]),max(a[1],b[1]); ix2,iy2 = min(a[2],b[2]),min(a[3],b[3])
        iw,ih = max(0,ix2-ix1),max(0,iy2-iy1); inter = iw*ih
        if inter==0: return 0.0
        return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    merged = []
    for box in boxes:
        hit = False
        for m in merged:
            if m[4]==box[4] and iou(m,box)>0.3:
                m[0]=min(m[0],box[0]);m[1]=min(m[1],box[1]);m[2]=max(m[2],box[2]);m[3]=max(m[3],box[3])
                hit=True; break
        if not hit: merged.append(box[:])
    for x1,y1,x2,y2,is_remove in merged:
        color=(0,0,220) if is_remove else (0,200,0); label="REMOVE" if is_remove else "KEEP"
        cv2.rectangle(preview,(x1,y1),(x2,y2),color,3)
        ty = y1-10 if y1-10>20 else y1+28
        cv2.putText(preview,label,(x1+4,ty),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2,cv2.LINE_AA)
    return preview


# ── 17. 디버그 영상 + 로그 생성 (v10_J) ─────────────────────────────────────────
def build_debug(frames, masks_raw_dir, refined, shadow_masks,
                fps, w, h, out_video, out_log, grouped, remove_ids):
    """
    디버그 영상: 3색 오버레이 (노랑=raw, 초록=refined, 빨강=shadow).
    좌상단 픽셀 통계 박스, 우상단 색상 범례.
    """
    n = len(frames)
    raw_files = sorted(Path(masks_raw_dir).glob("*.png"))
    raws = [(cv2.imread(str(raw_files[i]),0) if i<len(raw_files) else None) for i in range(n)]
    raws = [r if r is not None else np.zeros((h,w),np.uint8) for r in raws]
    raw_bool=[r>0 for r in raws]; refined_bool=[r>0 for r in refined]; shadow_bool=[s>0 for s in shadow_masks]
    keep_ids = [g for g in sorted(grouped["group_id"].unique()) if g not in remove_ids]
    log_lines = [
        "=== SceneEraser v11 디버그 로그 ===",
        f"영상: {w}x{h}, {n} frames, {fps:.1f} fps",
        f"감지된 그룹: {grouped['group_id'].nunique()}개",
        f"REMOVE group_ids: {remove_ids}",
        f"KEEP group_ids: {keep_ids}", "",
        f"{'frame':>6} | {'raw':>8} | {'refined':>8} | {'subtracted':>10} | {'shadow':>8} | {'shadow_add':>10}",
        "-"*70,
    ]
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    legend = [("YELLOW: raw mask (before refine)",(0,255,255)),
              ("GREEN : refined (person mask)",   (0,255,0)),
              ("RED   : shadow (final removed)",  (0,0,255))]
    for i in range(n):
        dbg = frames[i].copy()
        rw,rf,sh = raw_bool[i],refined_bool[i],shadow_bool[i]
        subtracted = rw & ~rf; shadow_add = sh & ~rf
        if rw.any():
            y=np.zeros_like(dbg);y[:]=(0,255,255);dbg[rw]=cv2.addWeighted(dbg,0.5,y,0.5,0)[rw]
        if rf.any():
            g=np.zeros_like(dbg);g[:]=(0,255,0);dbg[rf]=cv2.addWeighted(dbg,0.5,g,0.5,0)[rf]
        if sh.any():
            r=np.zeros_like(dbg);r[:]=(0,0,255);dbg[sh]=cv2.addWeighted(dbg,0.55,r,0.45,0)[sh]
        info_text = [f"frame: {i}/{n-1}",f"raw       : {int(rw.sum()):>8}",
                     f"refined   : {int(rf.sum()):>8}",f"subtracted: {int(subtracted.sum()):>8}",
                     f"shadow    : {int(sh.sum()):>8}",f"shadow add: {int(shadow_add.sum()):>8}"]
        ov=dbg.copy();cv2.rectangle(ov,(0,0),(300,175),(0,0,0),-1);cv2.addWeighted(ov,0.6,dbg,0.4,0,dbg)
        for k,line in enumerate(info_text):
            cv2.putText(dbg,line,(10,25+k*24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        lx,ly = w-330,5; ov2=dbg.copy()
        cv2.rectangle(ov2,(lx,ly),(w-5,ly+90),(0,0,0),-1);cv2.addWeighted(ov2,0.6,dbg,0.4,0,dbg)
        for k,(lbl,col) in enumerate(legend):
            cv2.rectangle(dbg,(lx+8,ly+10+k*26),(lx+26,ly+24+k*26),col,-1)
            cv2.putText(dbg,lbl,(lx+34,ly+23+k*26),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1,cv2.LINE_AA)
        writer.write(dbg)
        log_lines.append(f"{i:>6} | {int(rw.sum()):>8} | {int(rf.sum()):>8} "
                         f"| {int(subtracted.sum()):>10} | {int(sh.sum()):>8} | {int(shadow_add.sum()):>10}")
    writer.release()
    log_lines += ["","=== 요약 통계 ===",
                  f"평균 raw px     : {np.mean([m.sum() for m in raw_bool]):.0f}",
                  f"평균 refined px : {np.mean([m.sum() for m in refined_bool]):.0f}",
                  f"평균 shadow px  : {np.mean([m.sum() for m in shadow_bool]):.0f}"]
    Path(out_log).write_text("\n".join(log_lines), encoding="utf-8")


# ── 18. main entry point ────────────────────────────────────────────────────────
def run_pipeline(src, top_k=2, mask_mode="auto", composite_mode="A",
                 sam2_ckpt=None, tight_mask=False, shadow_off=False,
                 shadow_cap=False, debug=False, lum_match=True, lum_thresh=10.0,
                 progress=None):
    """
    composite_mode: "A"(다른프레임우선) / "B"(plate-only) / "C"(5프레임+plate)
    tight_mask: 마스크 확장량 축소.  shadow_off: 그림자 확장 생략.
    shadow_cap: 한 프레임 그림자 추가량 상한 적용.  debug: 디버그 영상/로그 생성.
    lum_match: 조명 유사 프레임 우선 차용 (v12, 고정 카메라 권장).
    lum_thresh: LAB L채널 평균 차이 허용 한계 (기본 10.0 ≈ 4%).
    """
    def _step(p, msg_ko, msg_en):
        if progress is not None: progress(p, desc=msg_ko)

    work       = Path(tempfile.mkdtemp(prefix="scene_eraser_"))
    src_p      = Path(src)
    pre_mp4    = work/"pre.mp4";    frames_dir = work/"frames"
    masks_raw  = work/"masks_raw";  masks_ref  = work/"masks_refined"
    silent_mp4 = work/"result_silent.mp4"
    result_mp4 = work/"result.mp4"; debug_mp4  = work/"debug.mp4"
    debug_log  = work/"debug_log.txt"; plate_png = work/"clean_plate.png"

    _step(0.05,"영상 전처리","Preprocessing video")
    info = preprocess_video(src_p, pre_mp4)
    if info["n"] == 0: return {"error":"유효한 프레임이 없습니다."}

    _step(0.10,"프레임 추출","Extracting frames")
    extract_frames(pre_mp4, frames_dir)

    _step(0.20,"사람 검출 및 추적","Detecting and tracking people")
    tracks = detect_track(pre_mp4)
    if tracks.empty: return {"error":"영상에서 사람을 감지하지 못했습니다."}

    _step(0.35,"그룹 분석","Analyzing groups")
    grouped    = merge_fragments(tracks)
    remove_ids = score_select(grouped, top_k=top_k, fw=info["w"], fh=info["h"])
    if not remove_ids: return {"error":"제거 대상을 결정하지 못했습니다."}
    remove_df  = grouped[grouped["group_id"].isin(remove_ids)].copy()

    _step(0.45,"마스크 생성 (SAM2)","Generating masks (SAM2)")
    generate_masks(frames_dir, remove_df, masks_raw, mode=mask_mode, sam2_ckpt=sam2_ckpt)

    _step(0.65,"마스크 다듬기","Refining masks")
    refined = refine_masks(frames_dir, masks_raw, masks_ref, tight=tight_mask)

    _step(0.75,"배경 plate 생성","Building background plate")
    fps_list    = _sorted_frames(frames_dir)
    frames      = [cv2.imread(str(fp)) for fp in fps_list]
    plate, res  = build_temporal_plate(frames, refined)
    clean_plate = inpaint_plate(plate, res)

    if shadow_off:
        _step(0.85,"그림자 확장 생략","Skipping shadow expansion")
        shadow_masks_out = refined
    else:
        _step(0.85,"그림자 영역 확장","Expanding shadow regions")
        shadow_masks_out = expand_shadow_masks(frames, refined, clean_plate, cap=shadow_cap)

    keep_masks = build_keep_masks(grouped, remove_ids, info["n"], info["h"], info["w"])

    _step(0.92,f"영상 합성 (모드 {composite_mode})",f"Compositing (mode {composite_mode})")
    composite_with_mode(frames, shadow_masks_out, clean_plate, info["fps"],
                        silent_mp4, info["w"], info["h"], mode=composite_mode,
                        keep_masks=keep_masks,
                        lum_match=lum_match, lum_thresh=lum_thresh)

    _step(0.96,"원본 오디오 합성","Muxing original audio")
    mux_audio(src_p, silent_mp4, result_mp4)

    _step(0.98,"결과 생성","Generating result")
    preview = build_preview(frames_dir, grouped, remove_ids)

    result = {
        "output_video": str(result_mp4), "preview": preview,
        "composite_mode": composite_mode, "remove_ids": remove_ids,
        "n_groups": int(grouped["group_id"].nunique()),
        # 디버그 셀용 중간 결과
        "_frames": frames, "_refined": refined, "_shadow_masks": shadow_masks_out,
        "_plate": plate, "_residual": res, "_clean_plate": clean_plate,
        "_grouped": grouped, "_remove_ids": remove_ids,
        "_fps_list": fps_list, "_work": work, "_info": info,
    }

    if debug:
        _step(0.99,"디버그 산출물 생성","Generating debug outputs")
        build_debug(frames, masks_raw, refined, shadow_masks_out,
                    info["fps"], info["w"], info["h"], debug_mp4, debug_log, grouped, remove_ids)
        cv2.imwrite(str(plate_png), clean_plate)
        result["debug_video"] = str(debug_mp4)
        result["debug_log"]   = str(debug_log)
        result["plate_image"] = str(plate_png)

    _step(1.0,"완료","Done")
    return result
