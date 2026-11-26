"""가짜 PoseFrame 생성기 (Sim).

모델·영상·Pi가 없어도 판단 로직과 메시지 흐름을 먼저 개발하기 위한 도구.
설계 문서의 "하드웨어 없이 가짜 구현(Sim)으로 먼저 개발" 원칙을 비전 쪽에도 적용한 것.

카메라 화면 기준 배치 (정규화 좌표)
  - 문틈 선: x = 0.72 세로선 (y 0.10 ~ 0.95), 오른쪽이 문 바깥
  - 사람은 기본적으로 x = 0.50 에 서 있고, 왼팔(화면 오른쪽 팔)이 문쪽 팔
  - 버튼 패널: x 0.58~0.66, y 0.35~0.50

시나리오 이름
  저장소 기본 9종 : normal, button_press, back_lean, point_gap, brace, dragged, slide, walk_out, leash_yank
  v0.4 8종       : v04_S1_walk ... v04_S9_dragged_feet_hidden (아래 'v0.4 합성 시나리오' 참고)
시나리오마다 문 설정이 다를 수 있으니 door_config(name) 으로 받는다.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List

import numpy as np

from .messages import Person, PetDetection, PoseFrame, round_kpts
from . import keypoints as K

DOOR_X = 0.72
SIM_DOOR_CONFIG = {
    "door": {"gap_p1": [DOOR_X, 0.10], "gap_p2": [DOOR_X, 0.95], "zone_width": 0.12,
             "button_panel": [0.58, 0.35, 0.66, 0.50], "inside_point": [0.4, 0.5]}
}

# 엉덩이 중점 기준 상대 좌표 (몸통 길이 T 단위, x 오른쪽, y 아래쪽). 카메라를 보고 선 자세.
_TEMPLATE = np.array([
    [0.00, -1.35],                       # nose
    [0.06, -1.40], [-0.06, -1.40],       # l_eye, r_eye
    [0.12, -1.37], [-0.12, -1.37],       # l_ear, r_ear
    [0.22, -1.00], [-0.22, -1.00],       # l_shoulder, r_shoulder
    [0.28, -0.55], [-0.28, -0.55],       # l_elbow, r_elbow
    [0.30, -0.10], [-0.30, -0.10],       # l_wrist, r_wrist
    [0.15, 0.00], [-0.15, 0.00],         # l_hip, r_hip
    [0.15, 0.60], [-0.15, 0.60],         # l_knee, r_knee
    [0.15, 1.20], [-0.15, 1.20],         # l_ankle, r_ankle
], dtype=np.float32)


class Body:
    """사람 하나의 자세를 조작하기 위한 도우미."""

    def __init__(self, hip=(0.50, 0.62), T=0.20, side_view: bool = False):
        self.hip = np.array(hip, dtype=np.float32)
        self.T = T
        self.rel = _TEMPLATE.copy()
        if side_view:                         # 옆모습: 좌우 관절이 거의 겹쳐 보임
            self.rel[:, 0] *= 0.25

    def lean(self, sin_away: float) -> None:
        """상체를 문 반대쪽(화면 왼쪽)으로 기울인다. 엉덩이를 축으로 회전."""
        ang = math.asin(max(-0.95, min(0.95, sin_away)))
        c, s = math.cos(ang), math.sin(ang)
        for i in range(0, 11):                # 머리·어깨·팔 (엉덩이 위)
            x, y = self.rel[i]
            # y는 위가 음수. 왼쪽(−x)으로 기울이려면 시계 반대 방향 회전
            self.rel[i] = [x * c + y * s, -x * s + y * c]

    def set_abs(self, idx: int, xy) -> None:
        self.rel[idx] = (np.asarray(xy, dtype=np.float32) - self.hip) / self.T

    def abs_xy(self, idx: int) -> np.ndarray:
        return self.hip + self.rel[idx] * self.T

    def reach(self, target_xy, arm: str = "left") -> None:
        """팔을 쭉 뻗어 손목을 target 에 둔다 (팔꿈치는 어깨-손목 사이 직선 위)."""
        s, e, w = K.ARMS[arm]
        sh = self.abs_xy(s)
        tgt = np.asarray(target_xy, dtype=np.float32)
        self.set_abs(w, tgt)
        self.set_abs(e, sh + (tgt - sh) * 0.5)

    def keypoints(self, rng: np.random.Generator, noise: float) -> List[List[float]]:
        xy = self.hip + self.rel * self.T + rng.normal(0, noise, self.rel.shape)
        conf = np.full((17, 1), 0.9, dtype=np.float32)
        return round_kpts(np.hstack([np.clip(xy, 0, 1), conf]).tolist())

    def bbox(self) -> List[float]:
        xy = self.hip + self.rel * self.T
        return [round(float(v), 4) for v in (*xy.min(0) - 0.02, *xy.max(0) + 0.02)]


def _smooth(t: float, t0: float, t1: float) -> float:
    """t0~t1 사이에서 0→1 로 부드럽게 변하는 값."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return u * u * (3 - 2 * u)


# ------------------------------------------------------------------ 시나리오
def _normal(t: float) -> Body:
    b = Body(hip=(0.50 + 0.005 * math.sin(t * 2), 0.62))
    return b


def _button_press(t: float) -> Body:
    b = Body(hip=(0.52, 0.62))
    k = _smooth(t, 0.5, 1.0)
    w0 = b.abs_xy(K.L_WRIST)
    b.reach(w0 + (np.array([0.62, 0.42]) - w0) * k)
    return b


def _back_lean(t: float) -> Body:
    """걸어가서 등을 문에 붙이고 선다. 팔은 몸 옆에 내린 채 (옆모습)."""
    k = _smooth(t, 0.3, 1.5)
    b = Body(hip=(0.50 + (DOOR_X - 0.03 - 0.50) * k, 0.62), side_view=True)
    ph = math.sin(t * 2 * math.pi * 1.6) * (1 if 0.3 < t < 1.5 else 0)
    b.rel[K.L_ANKLE, 0] += 0.25 * ph
    b.rel[K.R_ANKLE, 0] -= 0.25 * ph
    b.rel[K.L_WRIST, 0] -= 0.10                # 손은 몸보다 약간 앞(카 안쪽)
    b.rel[K.R_WRIST, 0] -= 0.10
    return b


def _point_gap(t: float) -> Body:
    b = Body(hip=(0.56, 0.62))
    k = _smooth(t, 0.3, 0.8)
    w0 = b.abs_xy(K.L_WRIST)
    tgt = np.array([DOOR_X - 0.01, 0.45])
    b.reach(w0 + (tgt - w0) * k)
    return b


def _brace(t: float) -> Body:
    b = Body(hip=(0.56, 0.62))
    b.lean(0.45 * _smooth(t, 0.8, 1.4))     # 손을 문틈에 둔 채 몸을 뒤로 젖힘
    b.reach([DOOR_X - 0.01, 0.47])
    return b


def _dragged(t: float) -> Body:
    """t<1.0: 손이 문틈에 끼임 → 1.0~1.6: 손이 먼저 문쪽으로 당겨짐 → 1.6~: 몸 전체가 끌려감."""
    pull = 0.10 * _smooth(t, 1.6, 2.6)                       # 몸 전체 이동
    b = Body(hip=(0.55 + pull, 0.62))
    b.lean(-0.25 * _smooth(t, 1.0, 1.6))                     # 상체가 문쪽으로 쏠림
    hand_x = DOOR_X - 0.01 + 0.03 * _smooth(t, 1.0, 1.6)      # 손이 문틈 안으로 먼저 빨려 들어감
    b.reach([hand_x, 0.47])
    return b                                                 # 발은 걷지 않고 함께 미끄러짐


def _slide(t: float) -> Body:
    b = Body(hip=(0.57, 0.62))
    y = 0.40 + 0.20 * _smooth(t, 1.0, 1.4)                   # 0.4초 동안 문틈을 따라 아래로
    b.reach([DOOR_X - 0.005, y])
    return b


def _walk_out(t: float) -> Body:
    """정상적으로 걸어서 내리는 사람 (오탐 확인용). 발이 번갈아 움직이고 팔을 흔든다."""
    x = 0.45 + 0.11 * _smooth(t, 0.5, 2.5)
    b = Body(hip=(x, 0.62))
    ph = math.sin(t * 2 * math.pi * 1.6) * (1 if 0.5 < t < 2.5 else 0)
    b.rel[K.L_ANKLE, 0] += 0.25 * ph
    b.rel[K.R_ANKLE, 0] -= 0.25 * ph
    b.rel[K.L_KNEE, 0] += 0.12 * ph
    b.rel[K.R_KNEE, 0] -= 0.12 * ph
    b.rel[K.L_WRIST, 0] -= 0.15 * ph
    b.rel[K.R_WRIST, 0] += 0.15 * ph
    return b


def _leash_yank(t: float) -> Body:
    """실제 Higgsfield 영상(목줄 끼임)을 흉내: 목줄 든 손이 문틈으로 끌려 올라감 →
    몸이 갑자기 문쪽으로 끌리며 한 발을 내딛음(비틀거림) → 무릎을 꿇고 두 손으로 버팀."""
    yank = _smooth(t, 2.0, 2.4)
    kneel = _smooth(t, 2.4, 3.0)
    b = Body(hip=(0.50 + 0.10 * yank, 0.62 + 0.08 * kneel))
    b.lean(-0.30 * yank + 0.15 * kneel)
    b.rel[K.L_ANKLE, 0] += 0.9 * yank * (1 - kneel)       # 끌려가며 내딛는 발
    b.rel[K.L_KNEE, 0] += 0.5 * yank
    b.rel[K.R_KNEE, 1] -= 0.3 * kneel                     # 한쪽 무릎 꿇음
    arm = _smooth(t, 1.2, 1.8)
    w0 = b.abs_xy(K.L_WRIST)
    b.reach(w0 + (np.array([DOOR_X - 0.01, 0.45 - 0.05 * kneel]) - w0) * arm)
    if kneel > 0:
        b.reach([DOOR_X - 0.02, 0.42], arm="right")
    return b


SCENARIOS: Dict[str, Callable[[float], Body]] = {
    "normal": _normal, "button_press": _button_press, "back_lean": _back_lean,
    "point_gap": _point_gap, "brace": _brace, "dragged": _dragged, "slide": _slide,
    "walk_out": _walk_out, "leash_yank": _leash_yank,
}

# 시나리오별 기대 결과 (evaluate.py, tests 에서 사용)
EXPECTED = {
    "normal": {"max_level": 0},
    "button_press": {"max_level": 0},
    "back_lean": {"max_level": 0, "flag": "leaning_on_door"},
    "point_gap": {"max_level": 1},
    "brace": {"max_level": 2},
    "dragged": {"max_level": 3, "danger_after": 1.0},
    "slide": {"max_level": 3, "danger_after": 1.0},
    "walk_out": {"max_level": 0},
    "leash_yank": {"max_level": 3, "danger_after": 2.0},
}


# ------------------------------------------------------- v0.4 합성 시나리오 (룰 엔진 v0.4 의 tests/synth.py)
# 원본은 픽셀 좌표(가로 1280 x 세로 1080 화면, 몸통 250px, 문틈 x=800 세로선, 카 안쪽은 왼쪽,
# 오른팔이 끼이는 팔). 좌표계를 그대로 옮기고 PoseFrame 에 img_w/img_h 를 넣어 해석이 화면비를 보정하게 한다.
_VW, _VH = 1280.0, 1080.0
_V_TORSO, _V_ARM, _V_SEAM_X = 250.0, 110.0, 800.0
V04_DOOR_CONFIG = {
    "door": {"gap_p1": [_V_SEAM_X / _VW, 100 / _VH], "gap_p2": [_V_SEAM_X / _VW, 1000 / _VH], "zone_width": 0.10,
             "button_panel": [660 / _VW, 200 / _VH, 770 / _VW, 290 / _VH], "inside_point": [400 / _VW, 550 / _VH]}
}
_V_PIVOT = np.array([_V_SEAM_X, 380.0])


def _v_elbow_ik(S, W, bend_sign=1.0):
    D = min(np.linalg.norm(W - S), 2 * _V_ARM * 0.999)
    mid = S + (W - S) / 2
    u = (W - S) / (np.linalg.norm(W - S) + 1e-9)
    perp = np.array([-u[1], u[0]]) * bend_sign
    return mid + perp * math.sqrt(max(_V_ARM ** 2 - (D / 2) ** 2, 0.0))


def _v_extend_toward(S, target, frac=0.995):
    u = (target - S) / np.linalg.norm(target - S)
    return S + u * (2 * _V_ARM * frac)


def _v_figure(hip_x, lean, r_wrist=None, gait=0.0, hip_y=600.0) -> np.ndarray:
    """(17, 3) 픽셀 키포인트. lean > 0 이면 문쪽(+x)으로 기욺."""
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    hip = np.array([hip_x, hip_y])
    sh = hip + np.array([lean, -_V_TORSO])
    S_l, S_r = sh + [0, -15], sh + [0, 15]
    k[5, :2], k[6, :2] = S_l, S_r
    k[11, :2], k[12, :2] = hip + [0, -15], hip + [0, 15]
    k[0, :2] = sh + [10, -60]
    k[1, :2], k[2, :2] = k[0, :2] + [-5, -8], k[0, :2] + [5, -8]
    k[3, :2], k[4, :2] = k[0, :2] + [-12, 0], k[0, :2] + [12, 0]
    an_l, an_r = np.array([hip_x + gait, 860.0]), np.array([hip_x - gait, 880.0])
    k[15, :2], k[16, :2] = an_l, an_r
    k[13, :2], k[14, :2] = (k[11, :2] + an_l) / 2, (k[12, :2] + an_r) / 2
    W_l = S_l + np.array([5.0, 200.0])
    k[9, :2], k[7, :2] = W_l, _v_elbow_ik(S_l, W_l, -1)
    if r_wrist is None:
        r_wrist = S_r + np.array([40.0, 170.0])
    k[10, :2], k[8, :2] = r_wrist, _v_elbow_ik(S_r, r_wrist, 1)
    return k


def _v_lerp(a, b, u):
    return a + (b - a) * np.clip(u, 0, 1)


def _v_S_r(hip_x, lean, hip_y=600.0):
    return np.array([hip_x + lean, hip_y - _V_TORSO + 15])


def _v_s1_walk(t):
    """문 쪽으로 걸어가 문에서 몸통 0.6배쯤에서 멈춤. 팔은 굽혀 흔듦."""
    hip_x, walking = _v_lerp(300, 650, t / 3.0), t < 3.0
    gait = 30 * math.sin(2 * math.pi * 2.0 * t) if walking else 0.0
    w = _v_S_r(hip_x, 10) + np.array([30 + (40 * math.sin(2 * math.pi * 2.0 * t) if walking else 0), 170])
    return _v_figure(hip_x, 10, w, gait)


def _v_s2_lean(t):
    """걸어가서 문에 손을 짚고 기댐."""
    hip_x = _v_lerp(300, 600, t / 2.5)
    gait = 30 * math.sin(2 * math.pi * 2.0 * t) if t < 2.5 else 0.0
    lean = _v_lerp(5, 40, (t - 2.5) / 0.5)
    relaxed = _v_S_r(hip_x, lean) + np.array([30, 170])
    w = relaxed if t < 2.5 else _v_lerp(relaxed, np.array([795.0, 400.0]), (t - 2.5) / 0.4)
    return _v_figure(hip_x, lean, w, gait)


def _v_s3_button(t):
    """걸어가서 문 옆 버튼을 팔 뻗어 누름."""
    hip_x = _v_lerp(300, 480, t / 2.0)
    gait = 30 * math.sin(2 * math.pi * 2.0 * t) if t < 2.0 else 0.0
    S = _v_S_r(hip_x, 5)
    relaxed = S + np.array([30, 170])
    w = _v_lerp(relaxed, _v_extend_toward(S, np.array([730.0, 240.0])), (t - 2.2) / 0.3) if 2.2 <= t <= 4.0 else relaxed
    return _v_figure(hip_x, 5, w, gait)


def _v_pull_phase(t, hip_x):
    """0~1s 가만히, 1~2s 줄이 팽팽해짐: 팔이 pivot 쪽으로 펴지고 몸은 버팀."""
    lean = _v_lerp(0, -45, (t - 1.0) / 1.0)
    S = _v_S_r(hip_x, lean)
    return lean, _v_lerp(S + np.array([40, 170]), _v_extend_toward(S, _V_PIVOT), (t - 1.0) / 1.0)


def _v_s5_self_pull(t):
    """줄이 끼었지만 스스로 당겨 버팀 (이김). 기대 CAUTION."""
    if t < 2.0:
        lean, w = _v_pull_phase(t, 520.0)
        return _v_figure(520.0, lean, w)
    hip_x = 520.0 - 20 * (t - 2.0)
    return _v_figure(hip_x, -45.0, _v_extend_toward(_v_S_r(hip_x, -45.0), _V_PIVOT))


def _v_s6_dragged(t):
    """줄이 끼어 버티다가, 걷지 않는데 몸이 pivot 쪽으로 끌려감. 손은 문틈에서 떨어져 있음."""
    if t < 2.0:
        lean, w = _v_pull_phase(t, 520.0)
        return _v_figure(520.0, lean, w)
    hip_x = min(520.0 + 100 * (t - 2.0), 625.0)
    S = _v_S_r(hip_x, -35.0)
    w = _v_extend_toward(S, _V_PIVOT) if np.linalg.norm(_V_PIVOT - S) > 2 * _V_ARM else _V_PIVOT.copy()
    return _v_figure(hip_x, -35.0, w)


def _v_s7_slide(t):
    """줄이 낀 점(pivot)이 문틈을 따라 아래로 미끄러짐 (카 상승), 손목이 따라감, 몸은 버팀. 손은 문에 안 닿음."""
    if t < 2.0:
        lean, w = _v_pull_phase(t, 520.0)
        return _v_figure(520.0, lean, w)
    p = _V_PIVOT + np.array([0.0, min(250 * (t - 2.0), 550)])
    return _v_figure(520.0, -45.0, _v_extend_toward(_v_S_r(520.0, -45.0), p))


def _v_s8_rigid_shift(t):
    """팔을 편 채 몸 전체가 문쪽으로 평행이동, 발 안 보임 (카메라 흔들림·화면 구성 문제). DANGER 면 안 됨."""
    hip_x = 450.0 + 90.0 * min(t, 3.0)
    k = _v_figure(hip_x, 0, _v_extend_toward(_v_S_r(hip_x, 0), np.array([_V_SEAM_X, 700.0])))
    k[[13, 14, 15, 16], 2] = 0.05
    return k


def _v_s9_dragged_feet_hidden(t):
    """S6 과 같지만 무릎·발목이 안 보임 (붐비는 카)."""
    k = _v_s6_dragged(t)
    k[[13, 14, 15, 16], 2] = 0.05
    return k


V04_SCENARIOS: Dict[str, Callable[[float], np.ndarray]] = {
    "v04_S1_walk": _v_s1_walk, "v04_S2_lean": _v_s2_lean, "v04_S3_button": _v_s3_button,
    "v04_S5_self_pull": _v_s5_self_pull, "v04_S6_dragged": _v_s6_dragged, "v04_S7_slide": _v_s7_slide,
    "v04_S8_rigid_shift": _v_s8_rigid_shift, "v04_S9_dragged_feet_hidden": _v_s9_dragged_feet_hidden,
}
SCENARIOS.update(V04_SCENARIOS)
EXPECTED.update({
    "v04_S1_walk": {"max_level": 0, "duration": 5.0},
    "v04_S2_lean": {"max_level": 0, "duration": 5.0},
    "v04_S3_button": {"max_level": 0, "duration": 5.0},
    "v04_S5_self_pull": {"max_level": 2, "duration": 5.0},
    "v04_S6_dragged": {"max_level": 3, "danger_after": 2.0, "duration": 5.0},
    "v04_S7_slide": {"max_level": 3, "danger_after": 2.0, "duration": 5.0,
                     "known_gap": "손이 문에 안 닿은 채 줄만 미끄러짐 → 가상 pivot(3단계) 필요"},
    "v04_S8_rigid_shift": {"max_level": 1, "duration": 4.0},
    "v04_S9_dragged_feet_hidden": {"max_level": 3, "danger_after": 2.0, "duration": 5.0},
})


def door_config(name: str) -> dict:
    """시나리오에 맞는 문 설정 (DoorGeometry.from_config 에 넣는다)."""
    return V04_DOOR_CONFIG if name in V04_SCENARIOS else SIM_DOOR_CONFIG


def generate(name: str, duration: float = 0.0, fps: float = 15.0, noise: float = 0.002,
             seed: int = 0, with_pet: bool = False, t0: float = 0.0) -> Iterator[PoseFrame]:
    """duration 0 이면 시나리오 기본 길이 (EXPECTED 의 duration, 없으면 3초)."""
    fn = SCENARIOS[name]
    duration = duration or EXPECTED.get(name, {}).get("duration", 3.0)
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        pets = []
        if with_pet:
            px = 0.60 + 0.10 * _smooth(t, 0.5, 1.5)
            pets.append(PetDetection("dog", 0.8, [px - 0.04, 0.85, px + 0.04, 0.95],
                                     abs(px - DOOR_X) <= 0.12))
        if name in V04_SCENARIOS:
            k = fn(t).copy()
            k[:, :2] /= [_VW, _VH]                                 # 픽셀 → 0~1
            k[:, :2] += rng.normal(0, noise, (17, 2))
            box = [float(v) for v in (*k[:, :2].min(0), *k[:, :2].max(0))]
            person = Person(1, 0.9, [round(v, 4) for v in box], round_kpts(k.tolist()))
            yield PoseFrame(ts=t0 + t, frame_id=i, persons=[person], pets=pets, source=f"sim:{name}",
                            img_w=int(_VW), img_h=int(_VH))
            continue
        body = fn(t)
        yield PoseFrame(ts=t0 + t, frame_id=i,
                        persons=[Person(1, 0.9, body.bbox(), body.keypoints(rng, noise))],
                        pets=pets, infer_ms=0.0, source=f"sim:{name}")
