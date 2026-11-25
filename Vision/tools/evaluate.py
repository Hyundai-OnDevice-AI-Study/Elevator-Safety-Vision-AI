"""시나리오 채점 — 설계 문서 '3차: 시나리오 테스트' / '비전 팀: 정확도 측정'.

두 가지 방식
  1) Sim 전체 채점 (모델 없이):
       python tools/evaluate.py --sim
  2) 실제 영상 채점: 먼저 vision_process --record 로 PoseFrame 을 jsonl 로 저장해 두고
       python tools/evaluate.py --labels config/scenarios/labels.json

labels.json 형식 (영상마다 한 줄)
  [{"name": "drag_01", "poses": "out/drag_01.jsonl", "vision_config": "config/vision.json",
    "expected_max_level": 3, "event_start_s": 1.6}, ...]
  - expected_max_level : 이 영상에서 나와야 할 최고 수준 (0 정상 / 1 주의 / 2 경계 / 3 위험)
  - event_start_s      : 실제 끌림이 시작된 시각(영상 시작 기준, 초). 있으면 감지 지연을 계산

지표
  - 시나리오 통과율 : 최고 수준이 기대값과 같은가
  - 미탐(FN)        : 위험(3)이어야 하는데 3이 안 나옴 → 가장 치명적
  - 오탐(FP)        : 위험이 아닌데 3이 나옴 → 엘리베이터가 불필요하게 멈춤
  - 감지 지연       : event_start_s 부터 처음 DANGER 까지 걸린 시간
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from door_drag.analysis.pose_features import PoseAnalyzer  # noqa: E402
from door_drag.messages import PoseFrame  # noqa: E402
from door_drag.transport import read_jsonl  # noqa: E402
from door_drag.vision.door_geometry import DoorGeometry  # noqa: E402


def score(frames, door, rules):
    a = PoseAnalyzer(door, rules.get("pose"))
    t0 = frames[0].ts if frames else 0.0
    mx, first, reasons, timeline = 0, None, set(), []
    for pf in frames:
        if not isinstance(pf, PoseFrame):
            continue
        r = a.update(pf, rules.get("pet"))
        timeline.append((round(pf.ts - t0, 2), r.level))
        mx = max(mx, r.level)
        for p in r.persons:
            reasons |= set(p.reasons)
        if r.level == 3 and first is None:
            first = pf.ts - t0
    return mx, first, sorted(reasons), timeline


def report(rows):
    print(f"\n{'scenario':16s} {'expect':>6s} {'got':>4s} {'first_DANGER':>12s} {'delay':>6s}  result  reasons")
    fn = fp = ok = 0
    delays = []
    for name, exp, got, first, ev, reasons in rows:
        res = "PASS" if got == exp else "FAIL"
        ok += res == "PASS"
        if exp == 3 and got < 3:
            fn += 1
            res += " (미탐)"
        if exp < 3 and got == 3:
            fp += 1
            res += " (오탐)"
        d = (first - ev) if (first is not None and ev is not None) else None
        if d is not None:
            delays.append(d)
        print(f"{name:16s} {exp:6d} {got:4d} {('%.2fs' % first) if first is not None else '-':>12s} "
              f"{('%.2fs' % d) if d is not None else '-':>6s}  {res:12s} {','.join(reasons)}")
    print(f"\n통과 {ok}/{len(rows)}  미탐 {fn}  오탐 {fp}" +
          (f"  평균 감지 지연 {sum(delays) / len(delays):.2f}s (최대 {max(delays):.2f}s)" if delays else ""))
    return ok == len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--labels")
    ap.add_argument("--rules", default="config/rules.json")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--noise", type=float, default=0.004)
    ap.add_argument("--timeline", action="store_true", help="프레임별 수준 출력")
    a = ap.parse_args()
    rules = json.load(open(a.rules, encoding="utf-8")) if os.path.exists(a.rules) else {}

    rows = []
    if a.sim:
        from door_drag.sim.fake_pose import EXPECTED, SCENARIOS, SIM_DOOR_CONFIG, generate
        door = DoorGeometry.from_config(SIM_DOOR_CONFIG)
        for name in SCENARIOS:
            frames = list(generate(name, fps=a.fps, noise=a.noise))
            mx, first, reasons, tl = score(frames, door, rules)
            ev = EXPECTED[name].get("danger_after")
            rows.append((name, EXPECTED[name]["max_level"], mx, first, ev, reasons))
            if a.timeline:
                print(name, tl)
    elif a.labels:
        for item in json.load(open(a.labels, encoding="utf-8")):
            cfg = json.load(open(item.get("vision_config", "config/vision.json"), encoding="utf-8"))
            if not os.path.exists(item["poses"]):
                print(f"[건너뜀] {item['name']}: {item['poses']} 이 없습니다. vision_process --record 로 먼저 만드세요.")
                continue
            frames = [f for f in read_jsonl(item["poses"]) if isinstance(f, PoseFrame)]
            n_person = sum(1 for f in frames if f.persons)
            if n_person == 0:
                print(f"[건너뜀] {item['name']}: 사람이 잡힌 프레임이 0개라 채점할 수 없습니다 "
                      f"(전체 {len(frames)}프레임). 포즈 추정이 제대로 돌았는지 확인하세요.")
                continue
            print(f"{item['name']}: {len(frames)}프레임 중 사람 검출 {n_person}프레임 ({n_person / len(frames):.0%})")
            mx, first, reasons, tl = score(frames, DoorGeometry.from_config(cfg), rules)
            rows.append((item["name"], item["expected_max_level"], mx, first, item.get("event_start_s"), reasons))
            if a.timeline:
                print(item["name"], tl)
    else:
        ap.error("--sim 또는 --labels 가 필요합니다")
    if not rows:
        print("\n채점할 데이터가 없습니다.")
        sys.exit(2)
    sys.exit(0 if report(rows) else 1)


if __name__ == "__main__":
    main()
