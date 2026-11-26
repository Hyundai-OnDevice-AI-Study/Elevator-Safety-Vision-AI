# Elevator-Safety-Vision-AI
엘레베이터 내부 끼임 사고 감지 — 줄·끈처럼 도어 센서가 못 잡는 끼임으로 사람이 끌려가는 상황을 감지·경고

## 폴더와 담당

```
common/     공유 (팀 합의로만 수정)   메시지 정의 · 전송(Queue/UDP/jsonl) · 문틈 기하 · 키포인트 번호 · 가짜 시나리오
Vision/     비전 팀                   영상 → YOLO-pose(ONNX) → 추적 → PoseFrame
Analysis/   해석·조건판별 팀          PoseFrame → 자세 규칙 → 수준(0~3)  (이후 state 상태 머신도 여기)
Control/    센싱·제어 팀              ESP32: 문·카 제어 / Pi: HX711 압력·경보 출력  (준비 중)
PJ1/        프로젝트 관리 문서
```

```
 Vision ──PoseFrame──▶ Analysis ──ActionCommand──▶ Control
                          ▲                          │
                          └── PressureSample · DeviceEvent
```

팀끼리는 `common/door_common/messages.py` 의 메시지로만 주고받는다. 메시지 필드를 바꾸면 `SCHEMA_VERSION` 을 올리고 모든 팀에 알린다.
지금 남아 있는 예외 하나: 비전 `--show` 화면이 판단 수준을 색으로 보여 주려고 해석 코드를 빌려 쓴다 (`vision_process.py` 주석 참고).

## 노트북에서 영상으로 결과 보기 (`run_video.py`)

영상 하나를 골라 **비전 → 해석 → 상태 머신**을 한 번에 돌리고, 화면에 결과를 겹쳐 본다.

```bash
# 설치 (화면이 필요하므로 opencv-python. headless 버전이 깔려 있으면 먼저 지운다)
pip uninstall -y opencv-python-headless
pip install numpy onnxruntime opencv-python

# 저장소 루트에서. 테스트할 영상 경로를 인자로 준다
python run_video.py Vision/videos/leash/leash_entrap_001.mp4
python run_video.py Vision/videos/leash/cam03_leash_drag.mp4 --events cam03
python run_video.py Vision/videos/leash/leash_entrap_001.mp4 \
       --events '[[0.0,"door","open"],[0.6,"door","closing"],[1.25,"door","closed"],[1.5,"car","moving"]]'
python run_video.py Vision/videos/leash/leash_entrap_003.mp4 --save out/leash_003.mp4   # 화면 없이 결과 영상 저장
```

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `video` (첫 인자) | 테스트할 영상 | (필수) |
| `--config` | 비전 설정: 모델, **문틈 좌표**(`door`) | `Vision/config/vision_cam03.json` |
| `--door`, `--car` | 시작할 때 문(`open/closing/closed`)·카(`stopped/moving`) 상태 | `closed`, `stopped` |
| `--events` | 시각별 문·카 이벤트 `[[초, kind, value], ...]`, 또는 `Analysis/config/scenarios/labels.json` 의 name | 없음 |
| `--realtime` | 처리가 느리면 프레임을 버림 (Pi 재현) | 끔 (모든 프레임을 영상 시간 기준으로 처리) |
| `--save` | 화면 대신 결과 영상(mp4) 저장 | 없음 |

**화면**: 사람 뼈대는 자세 수준 색(초록 NORMAL · 노랑 OBSERVE · 주황 CAUTION · 빨강 DANGER), 손목에 L/R 표시,
자홍 선 문틈, 하늘색 버튼 패널. 영상 아래 띠에 **상태 머신 상태**, 문·카 상태, 판단 근거, 최근 제어 명령 5개.
터미널에는 상태 변화와 명령이 시각과 함께 찍힌다.

**키**: `space` 일시정지/재생 · `d` 한 프레임(일시정지 중) · `o`/`c` 문 열림/닫힘 · `m` 카 이동↔정지 · `r` 관리자 리셋 · `s` 화면 png 저장 · `q` 종료

문·카 상태는 영상만으로는 알 수 없다. 영상을 보면서 키로 넣거나, `--events` 로 미리 준다.

**영상별 설정** (문틈 좌표가 맞아야 판단이 맞다)

| 영상 | `--config` |
|---|---|
| `leash/cam03_leash_drag`, `leash_entrap_001`, `003`, `008` | 기본값(cam03) 그대로 |
| `leash/leash_entrap_007` | 기본값으로 거의 맞음. 화면에서 자홍 선이 문틈에 겹치는지 확인 |
| `leash/leash_entrap_002`, `004`, `005`, `006_up`, `009`, `scarf/*` | 카메라 구도가 다름 → 설정 파일을 새로 만든다 (아래) |

```bash
cp Vision/config/vision_cam03.json Vision/config/vision_leash004.json
cd Vision && python tools/draw_door_line.py --video videos/leash/leash_entrap_004.mp4 --config config/vision_leash004.json && cd ..
python run_video.py Vision/videos/leash/leash_entrap_004.mp4 --config Vision/config/vision_leash004.json
```

## 바로 해 보기 (테스트)

```bash
pip install -r Vision/requirements.txt

cd Vision   && python -m pytest -q && python -m door_vision.vision_process --sim dragged --print && cd ..
cd Analysis && python -m pytest -q && python tools/evaluate.py --sim && cd ..
cd common   && python -m pytest -q && cd ..
```

각 팀 코드는 **자기 폴더에서** 실행한다. 설정 파일 안의 상대 경로도 그 폴더 기준이다. 따로 설치(pip install -e)할 필요 없이 `common/` 을 자동으로 찾는다.
