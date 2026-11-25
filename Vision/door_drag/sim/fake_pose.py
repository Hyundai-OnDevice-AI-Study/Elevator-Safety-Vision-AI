"""가짜 PoseFrame 생성기 (Sim).

모델·영상·Pi가 없어도 판단 로직과 메시지 흐름을 먼저 개발하기 위한 도구.
설계 문서의 "하드웨어 없이 가짜 구현(Sim)으로 먼저 개발" 원칙을 비전 쪽에도 적용한 것.

카메라 화면 기준 배치 (정규화 좌표)
  - 문틈 선: x = 0.72 세로선 (y 0.10 ~ 0.95), 오른쪽이 문 바깥
  - 사람은 기본적으로 x = 0.50 에 서 있고, 왼팔(화면 오른쪽 팔)이 문쪽 팔
  - 버튼 패널: x 0.58~0.66, y 0.35~0.50

시나리오 이름: normal, button_press, back_lean, point_gap, brace, dragged, slide, walk_out
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterator, List

import numpy as np

from ..messages import Person, PetDetection, PoseFrame, round_kpts
from ..vision import keypoints as K

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


def generate(name: str, duration: float = 3.0, fps: float = 15.0, noise: float = 0.002,
             seed: int = 0, with_pet: bool = False, t0: float = 0.0) -> Iterator[PoseFrame]:
    fn = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    n = int(duration * fps)
    for i in range(n):
        t = i / fps
        body = fn(t)
        pets = []
        if with_pet:
            px = 0.60 + 0.10 * _smooth(t, 0.5, 1.5)
            pets.append(PetDetection("dog", 0.8, [px - 0.04, 0.85, px + 0.04, 0.95],
                                     abs(px - DOOR_X) <= 0.12))
        yield PoseFrame(ts=t0 + t, frame_id=i,
                        persons=[Person(1, 0.9, body.bbox(), body.keypoints(rng, noise))],
                        pets=pets, infer_ms=0.0, source=f"sim:{name}")
