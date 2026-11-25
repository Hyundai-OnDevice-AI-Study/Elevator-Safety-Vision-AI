"""One Euro Filter — 키포인트 떨림 제거.

YOLO-pose 키포인트는 사람이 가만히 있어도 프레임마다 몇 픽셀씩 흔들린다.
평균 필터를 쓰면 떨림은 줄지만 빠른 움직임(=끌림!)까지 늦게 따라간다.
One Euro Filter는 '느릴 때는 강하게, 빠를 때는 약하게' 평활화해서 두 문제를 같이 줄인다.
  - min_cutoff 를 낮추면: 정지 상태 떨림이 더 줄어듦 (대신 느린 동작 지연↑)
  - beta 를 높이면: 빠른 동작을 더 빨리 따라감 (대신 빠를 때 떨림↑)
참고: Casiez et al., "1€ Filter", CHI 2012.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None or self.t_prev is None or t <= self.t_prev:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        a_d = _alpha(np.full_like(x, self.d_cutoff), dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class KeypointSmoother:
    """(17, 3) 키포인트 배열 중 x, y 만 평활화. 신뢰도 낮은 점은 이전 값 유지 안 함(그대로 통과)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5):
        self.f = OneEuroFilter(min_cutoff, beta)

    def __call__(self, kp: np.ndarray, t: float) -> np.ndarray:
        out = kp.copy()
        out[:, :2] = self.f(kp[:, :2], t)
        return out
