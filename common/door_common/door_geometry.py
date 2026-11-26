"""문틈 기준선과 관심 영역(ROI).

카메라는 카 안 천장 쪽에서 문을 바라본다고 가정한다.
문틈(두 문짝이 만나는 선 또는 문짝과 문틀 사이)을 영상 위의 선분 p1→p2 로 정의하고,
모든 자세 판단은 이 선을 기준으로 한 거리·방향으로 계산한다.

좌표는 0~1 정규화. Vision/tools/draw_door_line.py 로 영상에서 클릭해 vision*.json 의 "door" 에 저장한다.

영역 (v0.4 캘리브레이션에서 가져옴)
  button_panel : 사각형 [x1, y1, x2, y2]. 손이 머물면 버튼 누르기로 보고 약한 단계만 낮춘다
  exclude      : 다각형 목록. button_panel 과 같은 역할 (패널이 기울어져 보이거나 여러 개일 때)
  ignore       : 다각형 목록. 사람의 발 위치가 이 안이면 판단에서 뺀다 (거울·유리에 비친 사람)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DoorGeometry:
    p1: Tuple[float, float]                 # 문틈 선 위쪽 끝
    p2: Tuple[float, float]                 # 문틈 선 아래쪽 끝
    zone_width: float = 0.12                # 문 주변 영역 폭(정규화). 반려동물 판정용
    button_panel: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x1,y1,x2,y2
    inside_point: Tuple[float, float] = (0.5, 0.5)   # 카 안쪽에 있는 아무 점 (법선 부호 결정용)
    exclude: List[List[Tuple[float, float]]] = field(default_factory=list)   # 버튼 패널 다각형들
    ignore: List[List[Tuple[float, float]]] = field(default_factory=list)    # 거울 등 무시 영역 다각형들

    @staticmethod
    def from_config(cfg: dict) -> "DoorGeometry":
        d = cfg["door"]
        poly = lambda key: [[tuple(p) for p in pg] for pg in d.get(key, []) if len(pg) >= 3]
        return DoorGeometry(tuple(d["gap_p1"]), tuple(d["gap_p2"]), d.get("zone_width", 0.12),
                            tuple(d.get("button_panel", [0, 0, 0, 0])),
                            tuple(d.get("inside_point", [0.5, 0.5])), poly("exclude"), poly("ignore"))

    # ---- 기본 벡터
    @property
    def _a(self) -> np.ndarray:
        return np.asarray(self.p1, dtype=np.float32)

    @property
    def tangent(self) -> np.ndarray:
        """문틈 방향 단위벡터 (위→아래)."""
        v = np.asarray(self.p2, dtype=np.float32) - self._a
        return v / (np.linalg.norm(v) + 1e-9)

    @property
    def normal(self) -> np.ndarray:
        """문틈에 수직인 단위벡터. '카 안쪽(사람이 서 있는 쪽)'을 +로 둔다.

        설정의 inside_point(기본: 영상 중앙)가 카 안쪽이라고 보고 부호를 맞춘다.
        """
        t = self.tangent
        n = np.array([-t[1], t[0]], dtype=np.float32)
        if np.dot(np.asarray(self.inside_point, dtype=np.float32) - self._a, n) < 0:
            n = -n
        return n

    # ---- 점 단위 계산
    def signed_distance(self, xy: Sequence[float]) -> float:
        """문틈 선까지의 부호 있는 거리. +: 카 안쪽, 0: 문틈 위, -: 문 바깥쪽."""
        return float(np.dot(np.asarray(xy, dtype=np.float32) - self._a, self.normal))

    def along(self, xy: Sequence[float]) -> float:
        """문틈 선을 따라간 위치 (p1=0 기준)."""
        return float(np.dot(np.asarray(xy, dtype=np.float32) - self._a, self.tangent))

    def scaled(self, aspect: float) -> "DoorGeometry":
        """x 좌표에 화면비(w/h)를 곱한 '정사각 픽셀' 공간의 문 기하.

        정규화 좌표(0~1)는 16:9 영상에서 가로 1칸이 세로 1칸보다 1.78배 길다.
        몸통(세로)과 문틈까지 거리(가로)를 비교하려면 같은 단위로 맞춰야 한다.
        """
        if abs(aspect - 1.0) < 1e-6:
            return self
        sx = lambda p: (p[0] * aspect, p[1])
        x1, y1, x2, y2 = self.button_panel
        return DoorGeometry(sx(self.p1), sx(self.p2), self.zone_width * aspect,
                            (x1 * aspect, y1, x2 * aspect, y2), sx(self.inside_point),
                            [[sx(p) for p in pg] for pg in self.exclude],
                            [[sx(p) for p in pg] for pg in self.ignore])

    def in_button_panel(self, xy: Sequence[float]) -> bool:
        x1, y1, x2, y2 = self.button_panel
        if x1 < x2 and x1 <= xy[0] <= x2 and y1 <= xy[1] <= y2:
            return True
        return any(point_in_polygon(xy, pg) for pg in self.exclude)

    def in_ignore(self, kp: Sequence[Sequence[float]], conf: float = 0.35) -> bool:
        """사람의 바닥 위치(두 발목 중점, 없으면 가장 아래쪽 관절)가 무시 영역 안인가 (v0.4 in_ignore).

        거울에 비친 사람은 거울 면 안에 발이 있다. 실제 사람은 바닥에 서 있으므로 걸리지 않는다.
        """
        if not self.ignore:
            return False
        pt = floor_point(kp, conf)
        return pt is not None and any(point_in_polygon(pt, pg) for pg in self.ignore)

    def bbox_in_door_zone(self, bbox: Sequence[float]) -> bool:
        """박스의 아래쪽 중앙(발 위치)이나 중심이 문 주변 띠 안에 있는가."""
        x1, y1, x2, y2 = bbox
        for pt in ((0.5 * (x1 + x2), y2), (0.5 * (x1 + x2), 0.5 * (y1 + y2))):
            if abs(self.signed_distance(pt)) <= self.zone_width:
                return True
        return False


def floor_point(kp: Sequence[Sequence[float]], conf: float = 0.35) -> Optional[Tuple[float, float]]:
    """사람이 서 있는 바닥 위치: 두 발목(15, 16) 중 보이는 것의 평균, 없으면 가장 아래쪽(y 최대) 관절."""
    ank = [kp[i] for i in (15, 16) if kp[i][2] >= conf]
    if ank:
        return (sum(a[0] for a in ank) / len(ank), sum(a[1] for a in ank) / len(ank))
    ok = [p for p in kp if p[2] >= conf]
    if not ok:
        return None
    low = max(ok, key=lambda p: p[1])
    return (low[0], low[1])


def point_in_polygon(pt: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    """짝수-홀수 규칙(ray casting). cv2 없이 Pi·PC 어디서나 동작."""
    x, y = pt[0], pt[1]
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
