# 엘리베이터 도어 끌림 감지 — AI SW (비전 + 자세 해석)

설계 문서(2026-11-19, SW 설계 요약) 기준 **비전 프로세스**와 **자세 규칙 참고 구현**입니다.

```
영상 파일 ─▶ YOLO-pose(ONNX) ─▶ 추적 ─▶ [반려동물 탐지] ─▶ PoseFrame ──▶ 판단(해석팀)
                                                              └─▶ jsonl 기록 → evaluate.py 채점
```

## 폴더 구조

```
door_drag/
  messages.py            모듈 간 메시지 5종 (PoseFrame, PressureSample, DeviceEvent, StateChange, ActionCommand)
  transport.py           프로세스 간 전송 (multiprocessing.Queue / UDP / jsonl 기록)
  vision/
    onnx_backend.py      onnxruntime 실행, letterbox 전처리, NMS
    pose_estimator.py    YOLO-pose 출력 해석 (YOLOv8/11 형식, end-to-end 형식 모두)
    pet_detector.py      2차: 고양이·개 탐지 (COCO detect 모델)
    door_geometry.py     문틈 기준선, 버튼 패널, 문 주변 영역
    tracker.py           IoU 추적 (track_id)
    video_source.py      영상 파일을 '카메라처럼' 재생 (느리면 프레임 버림)
    vision_process.py    메인 루프 (별도 프로세스로 실행)
    draw.py              디버그 화면
  analysis/
    pose_features.py     설계 문서 3장 ① 자세 규칙 (주의/경계/위험)
    smoothing.py         One Euro 필터 (키포인트 떨림 제거)
  sim/fake_pose.py       가짜 PoseFrame 시나리오 8종 (모델 없이 개발)
config/
  vision.json            모델·영상·문틈 좌표
  rules.json             임계값 (코드에 숫자 넣지 않기)
tools/
  export_onnx.py         [PC] .pt → .onnx 변환
  quantize_int8.py       [PC] INT8 양자화 (선택)
  benchmark.py           [Pi] FPS·지연 측정
  draw_door_line.py      영상에서 문틈 선 클릭 → vision.json 저장
  evaluate.py            시나리오 채점 (통과율, 미탐, 오탐, 감지 지연)
tests/test_all.py        pytest (모델 없이 60개)
```

## 1. 모델 없이 바로 해 보기 (노트북에서 OK)

```bash
pip install -r requirements.txt
pytest -q                                          # 60 passed
python tools/evaluate.py --sim                     # 시나리오 8종 채점
python -m door_drag.vision.vision_process --sim dragged --print
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
33
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python tools/draw_door_line.py --video videos/sample.mp4          # 문틈 선 지정
python tools/benchmark.py --models models/yolo11n-pose_320.onnx models/yolo11n-pose_416.onnx \
                          --video videos/sample.mp4 --threads 2 3 4
python -m door_drag.vision.vision_process --source videos/sample.mp4 --show --print
python -m door_drag.vision.vision_process --source videos/drag_01.mp4 --record out/drag_01.jsonl
python tools/evaluate.py --labels config/scenarios/labels.json
```

## 판단(해석팀)과의 약속

- 비전은 **사람이 없어도, 영상이 끊겨도** PoseFrame 을 계속 보냄 (`status`: ok / no_frame / error).
  판단은 `rules.json`의 `vision_timeout_s` 동안 PoseFrame 이 안 오면 **고장 → 출발 금지**.
- 좌표는 전부 0~1 정규화. 시간은 `time.monotonic()` 초.
- `PoseAnalyzer` 결과는 **순간 근거**다. DANGER 유지(문이 열릴 때까지)와 압력·문 상태 결합은 상태 머신 몫.
- 문이 열리면 `PoseAnalyzer.reset()` 호출.

## 임계값 조정 순서

1. `vision_process --show` 로 키포인트가 문틈 근처에서 제대로 나오는지 눈으로 확인
2. `evaluate.py --timeline` 로 프레임별 수준 확인 → 실패한 시나리오의 `debug` 값(wrist_d, hip_v_to_door, lean …)을 보고 `rules.json` 수정
3. 수정 후 **모든** 시나리오를 다시 채점 (하나 고치면 다른 게 깨지는지 확인)
