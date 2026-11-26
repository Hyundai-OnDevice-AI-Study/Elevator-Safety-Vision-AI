"""시나리오 채점 — 설계 문서 '3차: 시나리오 테스트' / '비전 팀: 정확도 측정'.

두 가지 방식
  1) Sim 전체 채점 (모델 없이):
       python tools/evaluate.py --sim
  2) 실제 영상 채점: 먼저 vision_process --record 로 PoseFrame 을 jsonl 로 저장해 두고
       python tools/evaluate.py --labels config/scenarios/labels.json

labels.json 형식 (영상마다 한 줄)
  [{"name": "drag_01", "poses": "data/poses/drag_01.jsonl", "vision_config": "../Vision/config/vision.json",
    "expected_max_level": 3, "event_start_s": 1.6}, ...]
  - expected_max_level : 이 영상에서 나와야 할 최고 수준 (0 정상 / 1 주의 / 2 경계 / 3 위험)
  - event_start_s      : 실제 끌림이 시작된 시각(영상 시작 기준, 초). 있으면 감지 지연을 계산

지표 (통과 기준은 v0.4 run_offline 방식)
  - 통과 기준       : 기대 3(위험) → 최고 수준 3 / 기대 2(경계) → 정확히 2 / 기대 0·1 → 기대 이하
  - 미탐(FN)        : 위험(3)이어야 하는데 3이 안 나옴 → 가장 치명적
  - 오탐(FP)        : 위험이 아닌데 3이 나옴, 또는 사건 시작(event_start_s) 전에 3이 나옴
                      → 엘리베이터가 불필요하게 멈춤
  - 감지 지연       : event_start_s 부터 처음 DANGER 까지 걸린 시간
  - 단계별 첫 시각   : 처음 OBSERVE / CAUTION / DANGER 가 나온 시각
수준은 안정화(rules.json pose.stabilize)를 거친 값으로 채점한다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from door_analysis.pose_features import PoseAnalyzer  # noqa: E402  (common 경로도 여기서 잡힘)
from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_common.messages import PoseFrame  # noqa: E402
from door_common.transport import read_jsonl  # noqa: E402


def score(frames, door, rules):
    """반환: (최고 수준, {수준: 처음 나온 시각}, 근거 목록, [(시각, 수준)])"""
    a = PoseAnalyzer(door, rules.get("pose"))
    t0 = frames[0].ts if frames else 0.0
    mx, firsts, reasons, timeline = 0, {}, set(), []
    for pf in frames:
        if not isinstance(pf, PoseFrame):
            continue
        r = a.update(pf, rules.get("pet"))
        t = pf.ts - t0
        timeline.append((round(t, 2), r.level))
        mx = max(mx, r.level)
        for p in r.persons:
            reasons |= {x for x in p.reasons if not x.startswith("hold:")}
        for lv in range(1, r.level + 1):
            firsts.setdefault(lv, t)
    return mx, firsts, sorted(reasons), timeline


from door_analysis.scoring import passed  # noqa: E402


def report(rows):
    fmt = lambda v: ("%.2fs" % v) if v is not None else "-"
    print(f"\n{'scenario':26s} {'expect':>6s} {'got':>4s} {'OBSERVE':>8s} {'CAUTION':>8s} {'DANGER':>8s} "
          f"{'delay':>6s}  result  reasons")
    fn = fp = ok = 0
    delays = []
    for name, exp, got, firsts, ev, reasons, *gap in rows:
        first = firsts.get(3)
        early = first is not None and ev is not None and first < ev
        res = "PASS" if passed(exp, got) and not early else "FAIL"
        ok += res == "PASS"
        if exp == 3 and got < 3:
            fn += 1
            res += " (미탐)"
        if (exp < 3 and got == 3) or early:
            fp += 1
            res += " (사건 전 위험)" if early else " (오탐)"
        d = (first - ev) if (first is not None and ev is not None and not early) else None
        if d is not None:
            delays.append(d)
        if gap and gap[0] and res.startswith("FAIL"):
            res += " [알려진 한계]"
        print(f"{name:26s} {exp:6d} {got:4d} {fmt(firsts.get(1)):>8s} {fmt(firsts.get(2)):>8s} {fmt(first):>8s} "
              f"{fmt(d):>6s}  {res:12s} {','.join(reasons)}")
        if gap and gap[0] and res.startswith("FAIL"):
            print(f"{'':26s} └ {gap[0]}")
    known = sum(1 for r in rows if len(r) > 6 and r[6] and not passed(r[1], r[2]))
    print(f"\n통과 {ok}/{len(rows)}" + (f" (알려진 한계 {known}개 제외 시 {ok}/{len(rows) - known})" if known else "") +
          f"  미탐 {fn}  오탐 {fp}" +
          (f"  평균 감지 지연 {sum(delays) / len(delays):.2f}s (최대 {max(delays):.2f}s)" if delays else ""))
    return ok + known >= len(rows)


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
        from door_common.fake_pose import EXPECTED, SCENARIOS, door_config, generate
        for name in SCENARIOS:
            frames = list(generate(name, fps=a.fps, noise=a.noise))
            mx, firsts, reasons, tl = score(frames, DoorGeometry.from_config(door_config(name)), rules)
            ev = EXPECTED[name].get("danger_after")
            rows.append((name, EXPECTED[name]["max_level"], mx, firsts, ev, reasons, EXPECTED[name].get("known_gap")))
            if a.timeline:
                print(name, tl)
    elif a.labels:
        for item in json.load(open(a.labels, encoding="utf-8")):
            cfg = json.load(open(item.get("vision_config", "config/vision.json"), encoding="utf-8"))
            if not os.path.exists(item["poses"]):
                print(f"[건너뜀] {item['name']}: {item['poses']} 이 없습니다. 비전 팀이 vision_process --record 로 만들어 data/poses/ 에 넣어야 합니다.")
                continue
            frames = [f for f in read_jsonl(item["poses"]) if isinstance(f, PoseFrame)]
            n_person = sum(1 for f in frames if f.persons)
            if n_person == 0:
                print(f"[건너뜀] {item['name']}: 사람이 잡힌 프레임이 0개라 채점할 수 없습니다 "
                      f"(전체 {len(frames)}프레임). 포즈 추정이 제대로 돌았는지 확인하세요.")
                continue
            print(f"{item['name']}: {len(frames)}프레임 중 사람 검출 {n_person}프레임 ({n_person / len(frames):.0%})")
            mx, firsts, reasons, tl = score(frames, DoorGeometry.from_config(cfg), rules)
            rows.append((item["name"], item["expected_max_level"], mx, firsts, item.get("event_start_s"), reasons))
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
