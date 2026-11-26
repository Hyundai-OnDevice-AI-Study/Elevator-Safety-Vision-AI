# Analysis — 해석·조건판별 팀

비전이 보낸 **PoseFrame** 을 받아 자세 규칙으로 수준(0 NORMAL / 1 OBSERVE / 2 CAUTION / 3 DANGER)을 판단하고,
압력·문·카 상태와 합쳐 **안전 상태**를 정해 제어 명령(`ActionCommand`)을 낸다.
모델·영상 없이, 비전이 녹화한 jsonl 만으로 개발·채점할 수 있다.

출력은 사람마다 `level`(안정화된 수준)과 `raw_level`(그 프레임 규칙만으로 낸 순간 수준) 두 가지.
안정화는 수준별 최소 지속·유지 시간(`rules.json` 의 `stab_min_on_s`, `stab_hold_s`)으로 깜빡임을 없앤다.
'문이 열릴 때까지 DANGER 유지'(래치)는 상태 머신 몫이다.

```
door_analysis/
  pose_features.py     설계 문서 3장 ① 자세 규칙 (PoseAnalyzer)
  smoothing.py         One Euro 필터 (키포인트 떨림 제거, 관절별 신뢰도 게이트·끊김 후 초기화)
  stabilizer.py        수준 안정화 (최소 지속·유지 시간)
  scoring.py           채점 기준 (evaluate·stress·테스트 공용)
  state.py             안전 상태 머신 (6상태, 래치, 출발 보류, 명령)
  pipeline.py          PoseAnalyzer + 상태 머신 묶음 (메시지 넣으면 내보낼 메시지가 나옴)
config/
  rules.json           임계값 (pose · pet · pressure · vision_timeout_s). 코드에 숫자 넣지 않기
  scenarios/labels.json  채점용 영상 목록 (경로는 Analysis/ 기준)
data/poses/            비전이 녹화한 PoseFrame jsonl
tools/evaluate.py      시나리오 채점 (통과율, 미탐, 오탐, 단계별 첫 시각, 감지 지연)
tools/stress.py        노이즈 × FPS × 시드 조합 채점 (Pi 에서 FPS 가 떨어지면 어떻게 되는지)
tools/run_state.py     녹화 PoseFrame + 문·카 이벤트 재생 → 상태 변화·명령 출력
tests/test_analysis.py Sim 17종 + cam03 실영상 회귀 + 규칙별 단위 테스트
```

## 실행 (Analysis/ 폴더에서)

```bash
python -m pytest -q
python tools/evaluate.py --sim                                          # 가짜 시나리오 17종 채점
python tools/stress.py                                                  # fps 15/8/5 × 노이즈 3단계 × 시드 5개 (약 20초)
python tools/run_state.py --label cam03_leash_drag                      # 상태 머신 재생 (문·카 이벤트는 labels.json)
python tools/run_state.py --label cam03_leash_drag --reset-at 9.0       # 관리자 리셋까지
python tools/evaluate.py --labels config/scenarios/labels.json --timeline   # 실영상 채점, 프레임별 수준
```

문틈 좌표는 카메라 설치 정보라 비전 설정(`../Vision/config/vision*.json`)에 있다. `labels.json` 의 `vision_config` 가 그 파일을 가리킨다.

## 임계값 조정 순서

1. 비전 `--show` 로 키포인트가 문틈 근처에서 제대로 나오는지 눈으로 확인
2. `evaluate.py --timeline` 로 프레임별 수준 확인 → 실패한 시나리오의 `debug` 값(wrist_d, hip_v_to_door, lean …)을 보고 `rules.json` 수정
3. 수정 후 **모든** 시나리오를 다시 채점 (하나 고치면 다른 게 깨지는지 확인)

## DANGER 근거 4가지

| 근거 (`reasons`) | 내용 |
|---|---|
| `dragged_toward_door` | 걷지 않는데, 문틈 근처(몸통 0.35배 이내)에 있는 손을 따라 몸이 문쪽으로 끌려감 |
| `pulled_then_dragged` | 몸이 멈춘 채 팔이 문쪽으로 곧게 당겨진 뒤(줄이 팽팽해짐) 몸이 문쪽으로 움직임. 손이 멀어도 됨. 걷지 않아야 하고 손이 먼저 움직였어야 함 (목줄 쥐고 걸어 나가는 정상 동작 제외) |
| `hand_sliding_along_gap` | 문에 닿은 손이 문틈을 따라 2.0 torso/s 이상으로 미끄러짐 |
| `collapse_after_brace` | 최근 1초 안에 버팀(수준 2)이 있었고, 엉덩이가 발보다 문쪽으로 갑자기 쏠림 |

## 비자발성(끌림) 판단

- **걷기**: 문 방향 발목 간격이 +/- 로 바뀌는 횟수가 초당 1번(`gait_rate_max`)을 넘으면 걷는 중. 발이 안 보이면 '모름'
- **손 먼저**: 손목의 문쪽 속도가 엉덩이보다 0.1초 이상 먼저 올라감(`order_lead_s`), 또는 팔이 굽힘→폄으로 바뀐 뒤 몸이 움직임.
  발이 보이면 둘 중 하나, 발이 안 보이면 둘 다 필요
- 걷기 판정은 걸음을 세므로 걷기 시작한 첫 한두 걸음은 놓친다 → 손 먼저 근거가 그 틈을 막는다

## 가짜 시나리오 (`common/door_common/fake_pose.py`)

- 저장소 9종 + v0.4 합성 8종(`v04_*`). 시나리오마다 문 설정이 달라 `door_config(name)` 으로 받는다
- `v04_S7_slide`(손이 문에 안 닿은 채 줄만 미끄러짐)는 가상 pivot(3단계)이 있어야 잡을 수 있어 **알려진 한계**로 표시.
  테스트에서는 xfail, `evaluate.py --sim` 에서는 `[알려진 한계]`

## FPS 요구 (stress.py 결과)

15fps·8fps 에서는 알려진 한계를 빼고 전부 통과. **5fps 에서는 빠른 손 미끄러짐(`slide`, 0.4초짜리)을 자주 놓친다**
(2프레임밖에 안 잡힘). Pi 에서 해석에 들어오는 PoseFrame 이 **8fps 이상**이어야 한다.

## 안전 상태 머신 (`state.py`)

| 상태 | 들어가는 조건 | 나가는 명령 (상태가 바뀔 때 한 번) | 빠져나감 |
|---|---|---|---|
| DOOR_OPEN | 문 open | 부저·경광등 off, (보류 중이면) 출발 보류 해제 | 문 closing/closed |
| MONITORING | 문 닫힘, 근거 없음 | 부저·경광등 off | 근거 생기면 |
| WARNING | 자세 CAUTION · 압력 상승 · 문에 기댐 · 반려동물 문 앞 | 경광등 노랑, 부저 짧게, 음성 안내. 카가 서 있으면 출발 보류 (기댐만으로는 보류 안 함) | 근거 사라지고 1.5초 |
| DANGER_PRE | **카가 서 있을 때** 자세 DANGER, 또는 자세 CAUTION + 압력, 또는 반려동물 + 압력 | **문 다시 열기**, 출발 보류, 부저 연속, 빨강, 음성, 비상 전송 | **문이 열릴 때까지 유지** |
| DANGER_MOVING | **카가 움직일 때** 같은 근거 (또는 DANGER_PRE 중 카가 움직임) | **비상정지**, 부저 연속, 빨강, 음성, 비상 전송 | **관리자 리셋까지 유지** (`state.danger_moving_reset`) → 리셋 시 `emergency_reset` |
| FAULT | PoseFrame 이 `vision_timeout_s` 동안 없음 / status 오류 지속 / 장치 고장 이벤트 | 경광등 노랑 점멸, 음성, 고장 알림, 카가 서 있으면 출발 보류 | 원인 해소 |

우선순위 DANGER_MOVING > DANGER_PRE > FAULT > WARNING > MONITORING. 압력은 `pressure.threshold_g` 를 정하기 전까지 쓰지 않는다
(영점은 문이 열릴 때마다 다시 잡는다). `PoseAnalyzer` 결과는 순간 근거이고, 래치는 상태 머신이 맡는다.

센싱·제어 팀과 주고받는 이벤트·명령 이름은 `common/door_common/messages.py` 의 `DeviceEvent`·`ActionCommand` 주석에 정리했다
(`operator: reset`, `car: block_start/block_end/emergency_stop/emergency_reset`, `notify: emergency/fault` 등).
명령은 상태가 바뀔 때 한 번만 나간다. ESP32 도 `emergency_stop` 을 래치해서 `emergency_reset` 전까지 이동 명령을 무시하는 것을 권장.
