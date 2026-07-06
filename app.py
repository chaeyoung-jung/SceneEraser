"""SceneEraser v13 — Gradio 웹 UI.

파이프라인 로직은 pipeline.py에 있고, 이 파일은 UI와 실행 진입점만 담는다.
실행:
    python app.py                 # http://127.0.0.1:7860
    python app.py --port 8080
    python app.py --share         # 외부 공유 링크(Colab)
    python app.py --host 0.0.0.0  # 같은 네트워크의 다른 기기 접근
"""
from pipeline import *  # noqa: F401,F403  (run_pipeline, build_* 등 파이프라인 심볼)

import gradio as gr
import tempfile, traceback, cv2
from pathlib import Path

gr.close_all()
_debug_state = {}

CUSTOM_CSS = """
.gradio-container button.primary,
.gradio-container button.lg.primary,
.gradio-container .gr-button-primary,
.gradio-container button[class*="primary"] {
    background: #CD5C5C !important; background-color: #CD5C5C !important;
    background-image: none !important; border: none !important;
    color: #FFFFFF !important; transition: filter 0.2s ease !important;
}
.gradio-container button.primary:hover, .gradio-container button[class*="primary"]:hover {
    filter: brightness(1.1) !important; background: #CD5C5C !important;
}
.gradio-container button.primary:active, .gradio-container button[class*="primary"]:active {
    filter: brightness(0.9) !important;
}
.gradio-container input[type="range"] {
    -webkit-appearance: none !important; appearance: none !important;
    height: 6px !important; background: #CD5C5C !important;
    background-image: none !important; border-radius: 3px !important; outline: none !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important; appearance: none !important;
    width: 18px !important; height: 18px !important; border-radius: 50% !important;
    background: #CD5C5C !important; border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; cursor: pointer !important;
}
.gradio-container input[type="range"]::-moz-range-thumb {
    width: 18px !important; height: 18px !important; border-radius: 50% !important;
    background: #CD5C5C !important; border: 2px solid #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important; cursor: pointer !important;
}
input[type="radio"]:checked { accent-color: #CD5C5C !important; }
html.dark #app-header h2 { color: #CD5C5C !important; }
#app-header h2 { font-size: calc(1.5em + 12pt) !important; }
html.dark #debug-toggle, html.dark #debug-output { display: none !important; }
"""

DARK_TOGGLE_JS = """
() => {
    const html = document.documentElement;
    html.classList.toggle('dark');
    const isDark = html.classList.contains('dark');
    return [isDark ? '라이트모드' : '다크모드', isDark ? 'dark' : 'light'];
}
"""

def process(video_path, top_k, mask_mode, composite_label,
            tight_mask, shadow_off, shadow_cap, lum_match_on, debug_on, theme, sam2_ckpt_ui):
    global _debug_state
    _debug_state = {}

    if video_path is None:
        return None, None, None, "", None, "영상을 업로드해주세요."
    if int(top_k) == 0:
        return None, None, None, "", None, "제거할 객체 수를 1개 이상으로 설정해주세요."

    ckpt  = (sam2_ckpt_ui.strip() or None) if sam2_ckpt_ui else None
    debug = bool(debug_on) and (theme == "light")
    label_map = {
        "A · 안정성 우선":     "C",
        "B · 자연스러움 우선": "A",
        "C · plate only":     "B",
    }
    mode = label_map.get(composite_label, "C")
    mode_to_display = {"C": "A", "A": "B", "B": "C"}

    try:
        result = run_pipeline(
            video_path, top_k=int(top_k), mask_mode=mask_mode,
            composite_mode=mode, sam2_ckpt=ckpt,
            tight_mask=bool(tight_mask), shadow_off=bool(shadow_off),
            shadow_cap=bool(shadow_cap), debug=debug,
            lum_match=bool(lum_match_on),
        )
    except Exception as e:
        return None, None, None, "", None, f"오류: {e}\n{traceback.format_exc()}"

    if "error" in result:
        return None, None, None, "", None, result["error"]

    # 중간 결과 저장 (디버그 셀용)
    info = result.get("_info", {})
    _debug_state = {
        "frames":       result.get("_frames", []),
        "refined":      result.get("_refined", []),
        "shadow_masks": result.get("_shadow_masks", []),
        "plate":        result.get("_plate"),
        "residual":     result.get("_residual"),
        "clean_plate":  result.get("_clean_plate"),
        "grouped":      result.get("_grouped"),
        "remove_ids":   result.get("_remove_ids", []),
        "fps_list":     result.get("_fps_list", []),
        "work":         result.get("_work"),
        "info":         info,
    }

    # preview 이미지 파일로 저장
    preview_path = None
    if result.get("preview") is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, result["preview"])
        preview_path = tmp.name

    debug_video    = result.get("debug_video")
    plate_image    = result.get("plate_image")
    debug_log_text = ""
    log_path = result.get("debug_log")
    if log_path:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                debug_log_text = f.read()
        except Exception as e:
            debug_log_text = f"로그 읽기 실패: {e}"

    display_mode = mode_to_display.get(result["composite_mode"], result["composite_mode"])
    status = (f"감지된 그룹: {result['n_groups']}개 | "
              f"제거 대상: {result['remove_ids']} | "
              f"합성 모드: {display_mode}")
    return preview_path, result["output_video"], debug_video, debug_log_text, plate_image, status


with gr.Blocks(title="SceneEraser v11", css=CUSTOM_CSS,
               theme=gr.themes.Default(primary_hue="red")) as demo:
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
                "## SceneEraser v12\n"
                "영상에서 동적 객체(사람)를 자동 감지하고 제거합니다.  \n"
                "입력 영상은 **최대 1920×1080px / 30초**로 자동 제한됩니다.",
                elem_id="app-header"
            )
        with gr.Column(scale=1, min_width=140):
            dark_btn = gr.Button("라이트모드", size="sm", variant="secondary")

    theme_state = gr.Textbox(value="dark", visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            vi  = gr.Video(label="입력 영상 업로드", height=480)
            tk  = gr.Slider(0, 5, value=1, step=1, label="제거할 객체 수 (top-k)")
            mm  = gr.Radio(["sam2","bbox","auto"], value="sam2", label="마스크 모드",
                           info="sam2: 정밀 (기본) / bbox: 빠름 / auto: GPU 있으면 SAM2")
            cm  = gr.Radio(
                ["A · 안정성 우선", "B · 자연스러움 우선", "C · plate only"],
                value="A · 안정성 우선", label="합성 모드",
                info="A: 가까운 5프레임+plate fallback (기본) / B: 다른프레임 우선 / C: plate만",
            )
            lum_check = gr.Checkbox(value=True, label="조명 유사 프레임 우선 (v12)",
                                    info="밝기가 비슷한 프레임에서만 픽셀을 차용. 고정 카메라·동적 그림자에 권장.")
            with gr.Row():
                tight_check  = gr.Checkbox(value=False, label="마스크 과확장 축소",
                                           info="SAM2 권장. dilate 9→5, bottom 25→10.")
                soff_check   = gr.Checkbox(value=False, label="그림자 안 지움",
                                           info="사람만 제거, 그림자 유지.")
                scap_check   = gr.Checkbox(value=False, label="그림자 상한",
                                           info="추가량이 너무 크면 확장 스킵.")
            dbg_check = gr.Checkbox(value=False, label="디버그 모드",
                                    info="라이트모드 전환 후 ON 시 디버그 영상·로그·plate 생성.",
                                    elem_id="debug-toggle")
            ck  = gr.Textbox(label="SAM2 checkpoint 경로", value="/content/sam2_small.pt")
            btn = gr.Button("실행", variant="primary")
        with gr.Column(scale=1):
            vo  = gr.Video(label="출력 영상 (객체 제거)", height=480)
            pi  = gr.Image(label="감지 결과 (빨강=제거 / 초록=유지)", height=480)
            sb  = gr.Textbox(label="상태", interactive=False)

    with gr.Accordion("디버그 출력 (라이트모드 + 디버그 ON 시)", open=False, elem_id="debug-output"):
        gr.Markdown(
            "🟨 **노랑** = raw 마스크 (refine 전) / "
            "🟩 **초록** = refined (행인 최종 마스크) / "
            "🟥 **빨강** = shadow (실제 지워질 최종 영역)"
        )
        with gr.Row():
            dbg_vid = gr.Video(label="디버그 영상 (마스크 오버레이)", height=400)
            plt_img = gr.Image(label="배경 plate (clean plate)", height=400)
        dbg_log = gr.Textbox(label="디버그 로그 (프레임별 통계)",
                             lines=20, max_lines=300, interactive=False)

    btn.click(
        fn=process,
        inputs=[vi, tk, mm, cm, tight_check, soff_check, scap_check, lum_check, dbg_check, theme_state, ck],
        outputs=[pi, vo, dbg_vid, dbg_log, plt_img, sb],
    )
    dark_btn.click(fn=None, inputs=None, outputs=[dark_btn, theme_state], js=DARK_TOGGLE_JS)
    demo.load(fn=None, inputs=None, outputs=None,
              js="() => { document.documentElement.classList.add('dark'); }")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SceneEraser v13 Gradio 앱")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="외부 공유 링크 생성")
    args = parser.parse_args()

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
