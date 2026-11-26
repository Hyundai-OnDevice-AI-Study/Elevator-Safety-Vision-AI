"""디버그 화면 그리기 (뼈대, 문틈 선, 판단 수준)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from door_common.messages import PoseFrame
from door_common.door_geometry import DoorGeometry
from door_common.keypoints import SKELETON

LEVEL_COLORS = {0: (80, 200, 80), 1: (0, 200, 255), 2: (0, 140, 255), 3: (0, 0, 255)}


def draw_overlay(img: np.ndarray, pf: PoseFrame, door: DoorGeometry, kp_conf: float = 0.35,
                 assessment=None, fps: Optional[float] = None) -> np.ndarray:
    h, w = img.shape[:2]
    P = lambda xy: (int(xy[0] * w), int(xy[1] * h))
    out = img.copy()
    cv2.line(out, P(door.p1), P(door.p2), (255, 0, 255), 2)
    x1, y1, x2, y2 = door.button_panel
    if x1 < x2:
        cv2.rectangle(out, P((x1, y1)), P((x2, y2)), (255, 200, 0), 1)
    for pg, col in [(pg, (255, 200, 0)) for pg in door.exclude] + [(pg, (255, 128, 0)) for pg in door.ignore]:
        cv2.polylines(out, [np.array([P(p) for p in pg], np.int32)], True, col, 1)   # 하늘: 버튼, 주황: 무시
    levels = {a.track_id: a.level for a in assessment.persons} if assessment else {}
    for p in pf.persons:
        col = LEVEL_COLORS[levels.get(p.track_id, 0)]
        kp = p.keypoints
        for a, b in SKELETON:
            if kp[a][2] >= kp_conf and kp[b][2] >= kp_conf:
                cv2.line(out, P(kp[a]), P(kp[b]), col, 2)
        for x, y, c in kp:
            if c >= kp_conf:
                cv2.circle(out, P((x, y)), 3, (255, 255, 255), -1)
        cv2.putText(out, f"id{p.track_id} L{levels.get(p.track_id, 0)}", P(p.bbox[:2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    for pet in pf.pets:
        col = (0, 0, 255) if pet.in_door_zone else (200, 200, 0)
        cv2.rectangle(out, P(pet.bbox[:2]), P(pet.bbox[2:]), col, 2)
        cv2.putText(out, f"{pet.label} {pet.score:.2f}", P(pet.bbox[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    info = f"frame {pf.frame_id}  infer {pf.infer_ms:.0f}ms"
    if fps:
        info += f"  {fps:.1f} FPS"
    if assessment:
        info += f"  pose={assessment.level_name}"
    cv2.putText(out, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out
