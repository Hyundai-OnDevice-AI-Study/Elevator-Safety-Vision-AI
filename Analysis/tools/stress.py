"""스트레스 채점 — Sim 시나리오를 노이즈 × FPS × 시드 조합으로 돌려 통과율과 DANGER 지연을 본다 (v0.4 tests/stress.py).

Pi 에서 FPS 가 떨어지거나 키포인트가 더 흔들리면 결과가 어떻게 되는지 미리 확인하는 용도.

  python tools/stress.py                         # 기본: fps 15/8/5 × 노이즈 0.002/0.004/0.008 × 시드 5개
  python tools/stress.py --fps 15 10 --seeds 10
  python tools/stress.py --only v04_S6_dragged dragged

출력
  1) 시나리오별 통과율 (fps, 노이즈 조합마다)
  2) DANGER 시나리오의 평균 감지 지연
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # Analysis/
for _p in (HERE, os.path.join(os.path.dirname(HERE), "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from door_analysis.pose_features import PoseAnalyzer  # noqa: E402
from door_analysis.scoring import passed  # noqa: E402
from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_common.fake_pose import EXPECTED, SCENARIOS, door_config, generate  # noqa: E402


def run(name, fps, noise, seed, rules):
    a = PoseAnalyzer(DoorGeometry.from_config(door_config(name)), rules.get("pose"))
    mx, first = 0, None
    for pf in generate(name, fps=fps, noise=noise, seed=seed):
        lv = a.update(pf, rules.get("pet")).level
        mx = max(mx, lv)
        if lv == 3 and first is None:
            first = pf.ts
    return mx, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=os.path.join(HERE, "config", "rules.json"))
    ap.add_argument("--fps", type=float, nargs="+", default=[15, 8, 5])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.002, 0.004, 0.008])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", nargs="+", help="이 시나리오만")
    a = ap.parse_args()
    rules = json.load(open(a.rules, encoding="utf-8")) if os.path.exists(a.rules) else {}
    names = a.only or list(SCENARIOS)
    combos = [(f, n) for f in a.fps for n in a.noise]

    rate = {}
    lat = {}
    for name in names:
        exp = EXPECTED[name]
        for f, n in combos:
            ok, ds = 0, []
            for seed in range(a.seeds):
                mx, first = run(name, f, n, seed, rules)
                early = first is not None and "danger_after" in exp and first < exp["danger_after"]
                ok += passed(exp["max_level"], mx) and not early
                if exp["max_level"] == 3 and first is not None and not early:
                    ds.append(first - exp["danger_after"])
            rate[name, f, n] = ok / a.seeds
            if ds:
                lat[name, f, n] = float(np.mean(ds))

    head = "".join(f"{f'{f:g}fps/{n:g}':>13s}" for f, n in combos)
    print(f"\n통과율 (시드 {a.seeds}개)\n{'scenario':28s}{head}")
    for name in names:
        mark = " *" if "known_gap" in EXPECTED[name] else ""
        print(f"{name + mark:28s}" + "".join(f"{rate[name, f, n]:13.2f}" for f, n in combos))
    dangers = [x for x in names if EXPECTED[x]["max_level"] == 3]
    if dangers:
        print(f"\nDANGER 평균 감지 지연 (초)\n{'scenario':28s}{head}")
        for name in dangers:
            print(f"{name:28s}" + "".join(f"{lat[name, f, n]:13.2f}" if (name, f, n) in lat else f"{'-':>13s}"
                                          for f, n in combos))
    normal = [x for x in names if "known_gap" not in EXPECTED[x]]
    overall = np.mean([rate[x, f, n] for x in normal for f, n in combos])
    print(f"\n전체 통과율 (* 알려진 한계 제외): {overall:.3f}")


if __name__ == "__main__":
    main()
