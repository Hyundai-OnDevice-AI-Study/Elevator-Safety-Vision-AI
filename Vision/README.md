# Vision — 비전 팀

설계 문서(2026-11-19, SW 설계 요약) 기준 **비전 프로세스**입니다. 자세 해석은 `../Analysis`, 공유 코드는 `../common` 에 있습니다.

```
영상 파일 ─▶ YOLO-pose(ONNX) ─▶ 추적 ─▶ [반려동물 탐지] ─▶ PoseFrame ──▶ 해석(../Analysis)
                                                              └─▶ jsonl 기록 → ../Analysis/data/poses/
```

## 폴더 구조

```
door_vision/
  onnx_backend.py        onnxruntime 실행, letterbox 전처리, NMS
  pose_estimator.py      YOLO-pose 출력 해석 (YOLOv8/11 형식, end-to-end 형식 모두)
  pet_detector.py        2차: 고양이·개 탐지 (COCO detect 모델)
  tracker.py             IoU 추적 (track_id)
  video_source.py        영상 파일을 '카메라처럼' 재생 (느리면 프레임 버림)
  vision_process.py      메인 루프 (별도 프로세스로 실행)
  draw.py                디버그 화면
config/
  vision.json            모델·영상·문틈 좌표 (경로는 Vision/ 기준)
  vision_cam03.json      cam03 목줄 영상용
models/  videos/         ONNX 모델, 테스트 영상 (leash/, scarf/)
tools/
  export_onnx.py         [PC] .pt → .onnx 변환
  quantize_int8.py       [PC] INT8 양자화 (선택)
  benchmark.py           [Pi] FPS·지연 측정
  draw_door_line.py      영상에서 문틈 선 클릭 → vision.json 저장
tests/test_vision.py     pytest (모델 없이)

공유: ../common/door_common/  messages · transport · door_geometry · keypoints · fake_pose
```

## 1. 모델 없이 바로 해 보기 (노트북에서 OK)

```bash
pip install -r requirements.txt
python -m pytest -q
python -m door_vision.vision_process --sim dragged --print
```

## 2. 모델 준비 (PC)

```bash
pip install ultralytics onnx onnxslim
python tools/export_onnx.py --model yolo11n-pose.pt --imgsz 320 416
python tools/export_onnx.py --model yolo11n.pt --imgsz 320        # 2차 반려동물용
# models/*.onnx 를 Pi 로 복사
```

## 3. Raspberry Pi 4B 에서

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python tools/draw_door_line.py --video videos/sample.mp4          # 문틈 선 지정
python tools/benchmark.py --models models/yolo11n-pose_320.onnx models/yolo11n-pose_416.onnx \
                          --video videos/sample.mp4 --threads 2 3 4
python -m door_vision.vision_process --source videos/sample.mp4 --show --print
python -m door_vision.vision_process --config config/vision_cam03.json --show
# 해석 팀에 넘길 녹화 파일
python -m door_vision.vision_process --source videos/drag_01.mp4 --record ../Analysis/data/poses/drag_01.jsonl
```

## 해석 팀과의 약속

- 비전은 **사람이 없어도, 영상이 끊겨도** PoseFrame 을 계속 보냄 (`status`: ok / no_frame / error).
  해석은 `../Analysis/config/rules.json`의 `vision_timeout_s` 동안 PoseFrame 이 안 오면 **고장 → 출발 금지**.
- 좌표는 전부 0~1 정규화. 시간은 `time.monotonic()` 초.
- `PoseAnalyzer` 결과는 **순간 근거**다. DANGER 유지(문이 열릴 때까지)와 압력·문 상태 결합은 상태 머신 몫.
- 문이 열리면 `PoseAnalyzer.reset()` 호출.
- 새 영상을 녹화하면 `../Analysis/data/poses/` 에 넣고, 해석 팀에 `labels.json` 한 줄 추가를 요청한다.
