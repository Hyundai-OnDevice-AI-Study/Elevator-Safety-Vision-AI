# Analysis — 해석·조건판별 팀

비전이 보낸 **PoseFrame** 을 받아 자세 규칙으로 수준(0 NORMAL / 1 OBSERVE / 2 CAUTION / 3 DANGER)을 판단한다.
모델·영상 없이, 비전이 녹화한 jsonl 만으로 개발·채점할 수 있다.

```
door_analysis/
  pose_features.py     설계 문서 3장 ① 자세 규칙 (PoseAnalyzer)
  smoothing.py         One Euro 필터 (키포인트 떨림 제거)
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

## 앞으로

- `PoseAnalyzer` 결과는 **순간 근거**다. DANGER 유지(문이 열릴 때까지), 압력·문·카 상태 결합은 상태 머신(state, 예정)이 맡는다.
- 상태 이름은 `common/door_common/messages.py` 의 `StateChange` 에 정의된 6개
  (DOOR_OPEN · MONITORING · WARNING · DANGER_PRE · DANGER_MOVING · FAULT) 를 쓴다.
