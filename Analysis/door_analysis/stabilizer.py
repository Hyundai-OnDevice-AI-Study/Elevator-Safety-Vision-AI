"""수준 안정화 — 최소 지속 시간 + 유지 시간 (v0.4 Stabilizer 를 옮김).

규칙이 프레임마다 내는 '순간 수준(raw)'은 키포인트가 흔들리면 3 → 0 → 2 처럼 깜빡인다.
그대로 경보에 쓰면 부저가 켜졌다 꺼졌다 한다. 그래서 수준마다

  min_on_s : raw 가 이 수준 이상으로 이만큼 계속돼야 켜진다 (잠깐 튄 값 무시)
  hold_s   : 한 번 켜지면, raw 가 내려가도 이만큼은 유지한다 (깜빡임 제거)

를 적용한 '안정 수준(stable)'을 따로 낸다.

'문이 열릴 때까지 DANGER 유지'(래치)는 여기서 하지 않는다. 문·카 상태를 아는 상태 머신(state)의 몫이다.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence


class Stabilizer:
    def __init__(self, min_on_s: Sequence[float], hold_s: Sequence[float]):
        """min_on_s, hold_s: 수준 0~3 순서의 리스트 (0번은 쓰지 않음)."""
        self.levels = range(1, len(min_on_s))
        self.min_on = list(min_on_s)
        self.hold = list(hold_s)
        self.reset()

    def reset(self) -> None:
        self.start: Dict[int, Optional[float]] = {lv: None for lv in self.levels}   # raw 가 이 수준 이상이 된 시각
        self.armed: Dict[int, float] = {lv: -math.inf for lv in self.levels}         # 마지막으로 조건을 만족한 시각

    def update(self, t: float, raw: int) -> int:
        stable = 0
        for lv in self.levels:
            if raw >= lv:
                if self.start[lv] is None:
                    self.start[lv] = t
                if t - self.start[lv] >= self.min_on[lv] - 1e-9:
                    self.armed[lv] = t
            else:
                self.start[lv] = None
            if t - self.armed[lv] <= self.hold[lv]:
                stable = lv
        return stable
