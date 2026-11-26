# Elevator-Safety-Vision-AI
엘레베이터 내부 끼임 사고 감지 — 줄·끈처럼 도어 센서가 못 잡는 끼임으로 사람이 끌려가는 상황을 감지·경고

## 폴더와 담당

```
common/     공유 (팀 합의로만 수정)   메시지 정의 · 전송(Queue/UDP/jsonl) · 문틈 기하 · 키포인트 번호 · 가짜 시나리오
Vision/     비전 팀                   영상 → YOLO-pose(ONNX) → 추적 → PoseFrame
Analysis/   해석·조건판별 팀          PoseFrame → 자세 규칙 → 수준(0~3)  (이후 state 상태 머신도 여기)
Control/    센싱·제어 팀              HX711 압력, 헥사보드(ESP32) 문·카 제어  (준비 중)
PJ1/        프로젝트 관리 문서
```

```
 Vision ──PoseFrame──▶ Analysis ──ActionCommand──▶ Control
                          ▲                          │
                          └── PressureSample · DeviceEvent
```

팀끼리는 `common/door_common/messages.py` 의 메시지로만 주고받는다. 메시지 필드를 바꾸면 `SCHEMA_VERSION` 을 올리고 모든 팀에 알린다.
지금 남아 있는 예외 하나: 비전 `--show` 화면이 판단 수준을 색으로 보여 주려고 해석 코드를 빌려 쓴다 (`vision_process.py` 주석 참고).

## 바로 해 보기

```bash
pip install -r Vision/requirements.txt

cd Vision   && python -m pytest -q && python -m door_vision.vision_process --sim dragged --print && cd ..
cd Analysis && python -m pytest -q && python tools/evaluate.py --sim && cd ..
cd common   && python -m pytest -q && cd ..
```

각 팀 코드는 **자기 폴더에서** 실행한다. 설정 파일 안의 상대 경로도 그 폴더 기준이다. 따로 설치(pip install -e)할 필요 없이 `common/` 을 자동으로 찾는다.
