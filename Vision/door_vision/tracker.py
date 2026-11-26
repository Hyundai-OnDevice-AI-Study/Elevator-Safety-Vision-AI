"""아주 단순한 IoU 기반 추적기.

엘리베이터 카 안에는 사람이 많지 않으므로 무거운 추적기(ByteTrack 등)까지는 필요 없다.
프레임 사이에 같은 사람에게 같은 track_id 를 붙여 주는 것이 목적이다.
(자세 '변화'를 보려면 같은 사람의 이전 프레임과 비교해야 하기 때문)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


class IouTracker:
    def __init__(self, iou_thr: float = 0.3, max_missed: int = 10):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.tracks: Dict[int, dict] = {}
        self._next = 1

    def update(self, bboxes: List[np.ndarray]) -> List[int]:
        ids = [-1] * len(bboxes)
        pairs = []
        for tid, t in self.tracks.items():
            for i, b in enumerate(bboxes):
                v = iou(t["bbox"], b)
                if v >= self.iou_thr:
                    pairs.append((v, tid, i))
        used_t, used_d = set(), set()
        for v, tid, i in sorted(pairs, reverse=True):
            if tid in used_t or i in used_d:
                continue
            ids[i] = tid
            used_t.add(tid)
            used_d.add(i)
        for i, b in enumerate(bboxes):
            if ids[i] == -1:
                ids[i] = self._next
                self._next += 1
            self.tracks[ids[i]] = {"bbox": list(map(float, b)), "missed": 0}
        for tid in list(self.tracks):
            if tid not in ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return ids
