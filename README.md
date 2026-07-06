# SceneEraser

고정 카메라 영상에서 지나가는 행인은 자동 제거하고, 정적 인물(관람객 등)은 보존하는 파이프라인.

별도 마스크 지정 없이 영상만 입력하면 제거 대상을 자동 판별한다.

## 결과 영상

- [행인 1인 제거](https://sceneeraser.chaeyoung-jung.workers.dev/%5B1%5D.mp4) — AI 제작 영상
- [행인 2인 제거](https://sceneeraser.chaeyoung-jung.workers.dev/%5B2%5D.mp4) — AI 제작 영상

## 핵심 기능

- **KEEP / REMOVE 자동 분리** — 이동 궤적·위치·등장 패턴 점수화로 지울 행인과 보존할 인물 구분
- **깜빡임 없는 배경 복원** — 채우기 소스를 정적 배경 plate로 고정
- **Track 조각 병합** — occlusion 뒤 끊긴 track을 같은 사람으로 재결합 (ID swap 보정)
- **그림자 자동 처리** — HSV 그림자 검출 + connected-component 필터
- **전역 조명 보정** — LAB affine으로 밝기·화이트밸런스 drift 대응
- **GPU 없어도 동작** — SAM2(정밀) / bbox(경량) 두 마스크 모드, GPU 없으면 bbox 자동 fallback
- **디버그 인프라** — 마스크 단계별 오버레이 영상, 배경 plate 시각화, 프레임별 통계 로그

## 파이프라인

```
1. preprocess_video      해상도(≤1920×1080)·길이(≤30초) 제한
2. extract_frames        JPEG 프레임 추출
3. detect_track          YOLOv8n + ByteTrack 검출/추적
4. merge_fragments       끊긴 track 병합
5. score_select          점수화로 REMOVE 대상 선정
6. generate_masks        SAM2 또는 bbox 마스크 생성
7. refine_masks          morphology + 시간 보팅
8. build_temporal_plate  프레임별 중앙값으로 배경 plate 추정
9. inpaint_plate         plate 잔여 구멍 cv2.inpaint(TELEA)
10. expand_shadow_masks  HSV 그림자 검출로 마스크 확장
11. composite_with_mode  정적배경 → 다른 프레임 → plate 순으로 픽셀 채움
12. mux_audio            원본 오디오 재결합
```

제거 점수: 이동거리 35% · x축 이동폭 25% · 가장자리 등장 15% · 낮은 등장률 10% · 낮은 중심체류 10% · 적은 track 조각 5% → 상위 `top_k`개 선정.

## 설치

```bash
git clone <this-repo-url>
cd SceneEraser

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# torch/torchvision은 CUDA 버전에 맞춰 먼저 설치 (https://pytorch.org)
pip install -r requirements.txt

# SAM2 체크포인트 (small)
mkdir -p checkpoints
wget -O checkpoints/sam2_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

YOLOv8n 가중치는 첫 실행 시 자동 다운로드. GPU 없으면 마스크 모드에서 bbox 선택.

### Colab

```python
!pip install ultralytics gradio sam2 -q
!wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -O /content/sam2_small.pt
!python app.py --share
```

## 실행

```bash
python app.py                 # http://127.0.0.1:7860
python app.py --port 8080
python app.py --share         # 외부 공유 링크
python app.py --host 0.0.0.0  # 네트워크 내 다른 기기 접근
```

### 합성 모드

| UI 라벨 | 동작 |
|---|---|
| A · 안정성 우선 | 가까운 5프레임에서 차용 → plate fallback |
| B · 자연스러움 우선 | 전체 프레임에서 차용 → plate fallback |
| C · plate only | mask 영역 전부 배경 plate로 교체 |

## 가정

- 카메라 고정
- 행인은 한 방향으로 길게 이동 (좌↔우)
- KEEP 인물은 거의 안 움직임 (손 흔드는 정도는 OK)

가정이 깨지면 품질 저하 가능 (행인 정지, KEEP 대이동, 카메라 흔들림 등).

## 파일 구조

```
SceneEraser/
├── app.py            # Gradio UI + 실행 진입점
├── pipeline.py       # 영상 처리 파이프라인
├── requirements.txt
├── README.md
└── .gitignore
```

## 개발 히스토리 (v1 → v12)

과거 버전은 저장소에 포함하지 않고 변경 이력만 기록.

| 버전 | 핵심 변화 |
|---|---|
| v1 | CLI 스크립트 6개, 마스크 생성까지 (인페인팅 없음) |
| v2 | 단일 pipeline + Gradio, 중앙값 plate 단순 교체 |
| v3 | 시간적 복사 + feathering + 그림자 확장 + cv2.inpaint (깜빡임 잔존) |
| v4 | temporal median plate로 소스 고정 → 깜빡임 해결, bbox fallback, 합성 모드 |
| v9 | HSV 그림자 검출 + 정적배경 1순위 fill + KEEP 우회 |
| v10 | 진단 로그, 오디오 재결합, 마스크 tight/그림자 cap 옵션 |
| v11 | 디버그 인프라 (오버레이 영상·plate 시각화·통계 로그) |
| v12 | 조명 유사 프레임 우선 차용 + 전역 LAB affine plate 보정 |

깜빡임은 채우기 소스가 프레임마다 달라져서 발생. 소스를 전체 영상 중앙값 plate로 고정하면 블렌딩 없이도 사라짐 (v3 → v4).

## 개선 로드맵

현재 한계: 그림자·얼룩 잔재.

- **2-pass 배경 plate** — `build_temporal_plate`가 사람 마스크만 제외하고 중앙값을 계산해, 그림자가 지나간 바닥 픽셀이 plate에 섞여 오염됨. 1차 plate로 그림자 마스크를 먼저 구하고, 그림자까지 제외한 마스크로 plate를 재계산.
- **그림자 검출 민감도** — cap/tight OFF 기본화, `shadow_v_thresh` 하향, 탐색 범위 확대, 떨어진 지면 그림자용 connected-component 조건 완화.
- **경계 국소 보정** — 전역 LAB affine으로 못 잡는 국소 밝기차를 마스크 주변 밴드 국소 보정 또는 `cv2.seamlessClone`으로 처리.
- **잔여 영역 LaMa** — `cv2.inpaint`(TELEA)는 큰 영역을 뭉갬. clean_plate 1장에만 LaMa 1회 적용해 텍스처 복원.
- **optical-flow 기반 시간 복사** — nearest 프레임 대신 광류로 대응 픽셀 탐색, 미세 배경 움직임에 robust.
