"""해석 테스트: 자세 규칙(Sim 시나리오) + cam03 실영상 회귀.  (Analysis/ 에서 python -m pytest -q)"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("Analysis",):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))

import pytest  # noqa: E402

from door_analysis.pose_features import PoseAnalyzer  # noqa: E402
from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_common.fake_pose import EXPECTED, SCENARIOS, SIM_DOOR_CONFIG, generate  # noqa: E402
from door_common.messages import PoseFrame  # noqa: E402

DOOR = DoorGeometry.from_config(SIM_DOOR_CONFIG)


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("fps", [8, 15, 30])
@pytest.mark.parametrize("seed", [0, 1])
def test_scenarios(name, fps, seed):
    a = PoseAnalyzer(DOOR)
    levels, flags = [], set()
    for pf in generate(name, fps=fps, seed=seed, noise=0.004):
        r = a.update(pf)
        levels.append((pf.ts, r.level))
        for p in r.persons:
            flags |= {k for k, v in p.flags.items() if v}
    exp = EXPECTED[name]
    assert max(l for _, l in levels) == exp["max_level"]
    if "danger_after" in exp:      # 사건 전에는 DANGER 가 나오면 안 된다
        assert all(l < 3 for t, l in levels if t < exp["danger_after"])
    if "flag" in exp:
        assert exp["flag"] in flags


def test_reset_on_door_open():
    a = PoseAnalyzer(DOOR)
    for pf in generate("dragged"):
        a.update(pf)
    a.reset()
    assert a.tracks == {}


def test_pet_dwell():
    a = PoseAnalyzer(DOOR)
    res = [a.update(pf, {"dwell_s": 0.5}).pet_near_door for pf in generate("normal", with_pet=True)]
    assert not res[0] and res[-1]


def test_aspect_ratio_invariance():
    """같은 장면을 16:9 영상의 정규화 좌표로 보내도 판단이 같아야 한다."""
    import copy
    aspect = 1280 / 720
    d = copy.deepcopy(SIM_DOOR_CONFIG)
    for k in ("gap_p1", "gap_p2", "inside_point"):
        d["door"][k][0] /= aspect
    b = d["door"]["button_panel"]
    d["door"]["button_panel"] = [b[0] / aspect, b[1], b[2] / aspect, b[3]]
    d["door"]["zone_width"] /= aspect
    wide = DoorGeometry.from_config(d)
    for name in ("dragged", "walk_out", "brace", "leash_yank"):
        a1, a2 = PoseAnalyzer(DOOR), PoseAnalyzer(wide)
        l1, l2 = [], []
        for pf in generate(name, noise=0.003):
            l1.append(a1.update(pf).level)
            pf2 = copy.deepcopy(pf)
            pf2.img_w, pf2.img_h = 1280, 720
            for p in pf2.persons:
                for kp in p.keypoints:
                    kp[0] /= aspect
            l2.append(a2.update(pf2).level)
        assert max(l1) == max(l2), name


def test_real_cam03_leash_drag():
    """실제 영상(YOLO11n-pose 320)에서 뽑은 관절 데이터로 회귀 테스트.
    규칙을 바꿨을 때 이 영상의 결과가 나빠지면 바로 알 수 있다."""
    import json
    from door_common.transport import read_jsonl
    here = os.path.dirname(os.path.abspath(__file__))
    # 문틈 좌표는 비전 설정 파일에 있음 (카메라 설치 정보)
    cfg = json.load(open(os.path.join(_ROOT, "Vision", "config", "vision_cam03.json"), encoding="utf-8"))
    frames = [f for f in read_jsonl(os.path.join(here, "data", "cam03_leash_drag.jsonl")) if isinstance(f, PoseFrame)]
    a = PoseAnalyzer(DoorGeometry.from_config(cfg))
    t0 = frames[0].ts
    levels = [(f.ts - t0, a.update(f).level) for f in frames]
    assert all(l == 0 for t, l in levels if t < 3.9)                 # 사건 전(가만히 서서 휴대폰)에는 정상
    assert any(l >= 2 for t, l in levels if t < 5.5)                 # 목줄에 팔이 당겨지면 경계
    assert any(l == 3 for t, l in levels)                            # 끌려가면 위험
