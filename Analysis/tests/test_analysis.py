"""해석 테스트: 자세 규칙(Sim 시나리오) + cam03 실영상 회귀.  (Analysis/ 에서 python -m pytest -q)"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("Analysis",):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))

import math  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from door_analysis.pose_features import PoseAnalyzer  # noqa: E402
from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_analysis.scoring import passed  # noqa: E402
from door_common.fake_pose import EXPECTED, SCENARIOS, SIM_DOOR_CONFIG, door_config, generate  # noqa: E402
from door_common.messages import PoseFrame  # noqa: E402

DOOR = DoorGeometry.from_config(SIM_DOOR_CONFIG)


@pytest.mark.parametrize("name", [n if "known_gap" not in EXPECTED[n] else
                                  pytest.param(n, marks=pytest.mark.xfail(reason=EXPECTED[n]["known_gap"], strict=True))
                                  for n in SCENARIOS])
@pytest.mark.parametrize("fps", [8, 15, 30])
@pytest.mark.parametrize("seed", [0, 1])
def test_scenarios(name, fps, seed):
    """저장소 Sim 9종 + v0.4 합성 8종. 통과 기준은 evaluate.py 와 같은 scoring.passed."""
    a = PoseAnalyzer(DoorGeometry.from_config(door_config(name)))
    levels, flags = [], set()
    for pf in generate(name, fps=fps, seed=seed, noise=0.004):
        r = a.update(pf)
        levels.append((pf.ts, r.level))
        for p in r.persons:
            flags |= {k for k, v in p.flags.items() if v}
    exp = EXPECTED[name]
    assert passed(exp["max_level"], max(l for _, l in levels))
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
    first3 = next(t for t, l in levels if l == 3)
    assert first3 <= 6.2                                             # v0.4 반영 전 6.35s → 6.10s
    assert all(l == 3 for t, l in levels if t >= first3)             # 안정화: 끝까지 깜빡이지 않음


# ------------------------------------------------------------------ v0.4 에서 가져온 것들
from door_common import keypoints as K  # noqa: E402
from door_common.fake_pose import DOOR_X, Body, _smooth  # noqa: E402
from door_common.messages import Person  # noqa: E402


def _run(fn, duration=3.5, fps=15, seed=0, noise=0.003):
    """Body 를 돌려주는 함수 fn(t) 로 시나리오를 돌려 [(t, 안정 수준, 순간 수준, 근거)] 를 낸다."""
    rng = np.random.default_rng(seed)
    a = PoseAnalyzer(DOOR)
    out = []
    for i in range(int(duration * fps)):
        t = i / fps
        b = fn(t)
        r = a.update(PoseFrame(ts=t, frame_id=i, persons=[Person(1, 0.9, b.bbox(), b.keypoints(rng, noise))]))
        p = r.persons[0]
        out.append((t, r.level, p.raw_level, p.reasons))
    return out


def _pulled_far(t):
    """줄이 팽팽해지며 손이 먼저 문쪽으로 끌려감(0.3~0.8s, 몸은 그대로) + 몸은 뒤로 버팀
    → 1.5s 부터 몸이 문쪽으로 끌려감. 손은 문틈에서 몸통 0.5배 이상 떨어져 있어서
    예전 '손이 문틈 근처' 끌림 규칙으로는 못 잡던 경우 (v0.4 S6)."""
    b = Body(hip=(0.46 + 0.07 * _smooth(t, 1.5, 2.5), 0.62))
    b.lean(0.15 * _smooth(t, 0.3, 0.8))
    sh = b.abs_xy(K.L_SHOULDER)
    hang = b.abs_xy(K.L_WRIST)
    pull = _smooth(t, 0.3, 0.8)
    b.reach(hang + (sh + np.array([0.17, 0.02]) - hang) * pull)   # 내린 팔 → 문쪽으로 곧게 (손이 먼저)
    return b


def _walk_out_with_leash(t):
    """개가 앞에서 당기는 목줄을 쥐고 서 있다가(팔이 문쪽으로 곧게) 1.2s 부터 걸어서 내림 → 정상."""
    x = 0.45 + 0.12 * _smooth(t, 1.2, 3.2)
    b = Body(hip=(x, 0.62))
    ph = math.sin(t * 2 * math.pi * 1.6) * (1 if 1.2 < t < 3.2 else 0)
    b.rel[K.L_ANKLE, 0] += 0.25 * ph
    b.rel[K.R_ANKLE, 0] -= 0.25 * ph
    sh = b.abs_xy(K.L_SHOULDER)
    b.reach(sh + np.array([0.17, 0.03]))
    return b


def _rigid_shift(t):
    """팔을 문쪽으로 뻗은 채 몸 전체가 처음부터 같이 평행이동 (카메라 흔들림·그냥 다가감). 발 안 보임 (v0.4 S8)."""
    b = Body(hip=(0.44 + 0.05 * min(t, 2.0), 0.62))
    sh = b.abs_xy(K.L_SHOULDER)
    b.reach(sh + np.array([0.17, 0.03]))
    return b


def test_pulled_then_dragged_with_hand_far_from_gap():
    tl = _run(_pulled_far)
    assert all(l < 3 for t, l, _, _ in tl if t < 1.5)
    hits = [(t, rs) for t, l, _, rs in tl if l == 3]
    assert hits and hits[0][0] < 2.6
    assert any("pulled_then_dragged" in rs for _, rs in hits)


def test_walk_out_holding_leash_is_not_danger():
    for seed in range(3):
        assert max(l for _, l, _, _ in _run(_walk_out_with_leash, seed=seed)) < 3


def test_rigid_shift_with_hidden_feet_is_not_danger():
    for seed in range(3):
        rng = np.random.default_rng(seed)
        a = PoseAnalyzer(DOOR)
        mx = 0
        for i in range(int(3.0 * 15)):
            b = _rigid_shift(i / 15)
            kp = np.array(b.keypoints(rng, 0.003))
            kp[[K.L_KNEE, K.R_KNEE, K.L_ANKLE, K.R_ANKLE], 2] = 0.05      # 발이 안 보임
            mx = max(mx, a.update(PoseFrame(ts=i / 15, frame_id=i, persons=[Person(1, 0.9, b.bbox(), kp.tolist())])).level)
        assert mx < 3, seed


def test_stabilizer_min_on_and_hold():
    from door_analysis.stabilizer import Stabilizer
    s = Stabilizer([0, 0.2, 0.2, 0.0], [0, 1.0, 1.5, 2.0])
    assert s.update(0.0, 2) == 0          # 0.2초 안 됨
    assert s.update(0.1, 2) == 0
    assert s.update(0.25, 2) == 2         # 0.2초 넘음
    assert s.update(0.3, 0) == 2          # 내려가도 유지
    assert s.update(1.7, 0) == 2          # 마지막 만족(0.25) + 1.5 = 1.75 까지
    assert s.update(1.8, 0) == 0          # CAUTION(1.5)·OBSERVE(1.0) 유지 둘 다 끝남
    s.reset()
    assert s.update(5.0, 3) == 3          # DANGER 는 최소 지속 0 → 바로
    assert s.update(6.9, 0) == 3 and s.update(7.1, 0) <= 2


def test_smoother_ignores_low_conf_and_restarts_after_gap():
    from door_analysis.smoothing import KeypointSmoother
    sm = KeypointSmoother(1.5, 5.0, min_conf=0.35, max_gap_s=0.3)
    kp = np.zeros((17, 3), np.float32)
    kp[:, :2], kp[:, 2] = 0.5, 0.9
    for i in range(5):
        sm(kp, i * 0.066)
    bad = kp.copy()
    bad[9] = [0.95, 0.95, 0.1]                     # 튀었지만 신뢰도 낮은 손목
    sm(bad, 0.4)
    assert np.allclose(sm(kp, 0.47)[9, :2], 0.5, atol=1e-3)     # 필터 상태가 오염되지 않음
    far = kp.copy()
    far[9, :2] = 0.8
    assert np.allclose(sm(far, 1.5)[9, :2], 0.8)  # 1초 끊긴 뒤 다시 나타나면 예전 값과 섞지 않음


def test_posture_needs_both_sides():
    from door_analysis.pose_features import measure
    b = Body(hip=(0.5, 0.62))
    kp = np.array(b.keypoints(np.random.default_rng(0), 0.0), np.float32)
    full = measure(kp, 0.0, DOOR, 0.35)
    assert full.lean is not None and full.d_hip is not None
    kp[K.R_HIP, 2] = 0.1                           # 엉덩이 한쪽이 안 보임
    half = measure(kp, 0.0, DOOR, 0.35)
    assert half is not None and half.d_hip is None and half.lean is None


def test_torso_length_uses_median():
    from collections import deque
    from door_analysis.pose_features import measure
    hist = deque(maxlen=31)
    rng = np.random.default_rng(0)
    for i in range(10):
        measure(np.array(Body(hip=(0.5, 0.62)).keypoints(rng, 0.0), np.float32), i * 0.066, DOOR, 0.35, hist)
    short = Body(hip=(0.5, 0.62), T=0.10)           # 한 프레임만 몸통이 절반으로 보임 (숙임)
    m = measure(np.array(short.keypoints(rng, 0.0), np.float32), 1.0, DOOR, 0.35, hist)
    assert abs(m.torso - 0.20) < 0.01


def test_person_in_ignore_zone_is_skipped():
    """거울에 비친 사람(발이 ignore 영역 안)은 판단하지 않는다 (v0.4 ignore)."""
    cfg = {"door": {**SIM_DOOR_CONFIG["door"], "ignore": [[(0.0, 0.7), (0.35, 0.7), (0.35, 1.0), (0.0, 1.0)]]}}
    a = PoseAnalyzer(DoorGeometry.from_config(cfg))
    mirror = next(generate("dragged", fps=15))
    kp = np.array(mirror.persons[0].keypoints)
    kp[:, 0] -= 0.30                                   # 화면 왼쪽(거울 영역)으로 옮김
    real = mirror.persons[0]
    pf = PoseFrame(ts=0.0, frame_id=0, persons=[Person(7, 0.9, [0, 0, 1, 1], kp.tolist()), real])
    r = a.update(pf)
    assert [p.track_id for p in r.persons] == [real.track_id]
