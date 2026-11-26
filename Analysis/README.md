# Analysis — 해석·조건판별 팀

비전이 보낸 **PoseFrame** 을 받아 자세 규칙으로 수준(0 NORMAL / 1 OBSERVE / 2 CAUTION / 3 DANGER)을 판단한다.
모델·영상 없이, 비전이 녹화한 jsonl 만으로 개발·채점할 수 있다.

출력은 사람마다 `level`(안정화된 수준)과 `raw_level`(그 프레임 규칙만으로 낸 순간 수준) 두 가지.
안정화는 수준별 최소 지속·유지 시간(`rules.json` 의 `stab_min_on_s`, `stab_hold_s`)으로 깜빡임을 없앤다.
'문이 열릴 때까지 DANGER 유지'(래치)는 상태 머신 몫이다.

```
door_analysis/
  pose_features.py     설계 문서 3장 ① 자세 규칙 (PoseAnalyzer)
  smoothing.py         One Euro 필터 (키포인트 떨림 제거, 관절별 신뢰도 게이트·끊김 후 초기화)
  stabilizer.py        수준 안정화 (최소 지속·유지 시간)
config/
  rules.json           임계값 (pose · pet · pressure · vision_timeout_s). 코드에 숫자 넣지 않기
  scenarios/labels.json  채점용 영상 목록 (경로는 Analysis/ 기준)
data/poses/            비전이 녹화한 PoseFrame jsonl
tools/evaluate.py      시나리오 채점 (통과율, 미탐, 오탐, 감지 지연)
tests/test_analysis.py Sim 시나리오 + cam03 실영상 회귀 (tests/data/)
```

## 실행 (Analysis/ 폴더에서)

```bash
python -m pytest -q
python tools/evaluate.py --sim                                          # 가짜 시나리오 채점
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
| `pulled_then_dragged` | 몸이 멈춘 채 팔이 문쪽으로 곧게 당겨진 뒤(줄이 팽팽해짐) 몸이 문쪽으로 움직임. 손이 멀어도 됨. 걷는 중이면 제외 (목줄 쥐고 걸어 나가는 정상 동작) |
| `hand_sliding_along_gap` | 문에 닿은 손이 문틈을 따라 2.0 torso/s 이상으로 미끄러짐 |
| `collapse_after_brace` | 최근 1초 안에 버팀(수준 2)이 있었고, 엉덩이가 발보다 문쪽으로 갑자기 쏠림 |

## 앞으로

- `PoseAnalyzer` 결과는 **순간 근거**다. DANGER 유지(문이 열릴 때까지), 압력·문·카 상태 결합은 상태 머신(state, 예정)이 맡는다.
- 상태 이름은 `common/door_common/messages.py` 의 `StateChange` 에 정의된 6개
  (DOOR_OPEN · MONITORING · WARNING · DANGER_PRE · DANGER_MOVING · FAULT) 를 쓴다.
