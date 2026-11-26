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
    """(17, 3) 키포인트의 x, y 를 관절마다 따로 평활화한다.

    v0.4 에서 가져온 두 가지:
      - 신뢰도 게이트: conf < min_conf 인 관절은 필터에 넣지 않는다 (튀는 값이 필터 상태를 오염시키지 않게).
        그 관절은 입력 그대로 내보내고, 어차피 measure() 에서 '없는 점'으로 버려진다.
      - 끊김 후 초기화: 어떤 관절이 max_gap_s 보다 오래 안 보였다가 다시 나타나면, 그 관절만
        필터를 새로 시작한다. 예전 위치와 이어 붙이면 가짜 '빠른 이동'이 생긴다.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0,
                 min_conf: float = 0.0, max_gap_s: float = 0.3):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.min_conf, self.max_gap_s = min_conf, max_gap_s
        self.x_prev: Optional[np.ndarray] = None      # (17, 2)
        self.dx_prev: Optional[np.ndarray] = None     # (17, 2)
        self.t_seen: Optional[np.ndarray] = None      # (17,) 마지막으로 필터에 넣은 시각

    def __call__(self, kp: np.ndarray, t: float) -> np.ndarray:
        kp = np.asarray(kp, dtype=np.float32)
        out = kp.copy()
        n = len(kp)
        if self.x_prev is None:
            self.x_prev = np.zeros((n, 2), np.float32)
            self.dx_prev = np.zeros((n, 2), np.float32)
            self.t_seen = np.full(n, -np.inf)
        for i in range(n):
            if kp[i, 2] < self.min_conf:
                continue                                   # 신뢰도 낮음: 상태 건드리지 않음
            x = kp[i, :2]
            dt = t - self.t_seen[i]
            if not (0.0 < dt <= self.max_gap_s):           # 처음 보거나, 오래 끊겼거나, 시간이 거꾸로
                self.x_prev[i], self.dx_prev[i] = x, 0.0
            else:
                dx = (x - self.x_prev[i]) / dt
                a_d = float(_alpha(np.float32(self.d_cutoff), dt))
                dx_hat = a_d * dx + (1 - a_d) * self.dx_prev[i]
                a = _alpha(self.min_cutoff + self.beta * np.abs(dx_hat), dt)
                self.x_prev[i] = a * x + (1 - a) * self.x_prev[i]
                self.dx_prev[i] = dx_hat
            self.t_seen[i] = t
            out[i, :2] = self.x_prev[i]
        return out
