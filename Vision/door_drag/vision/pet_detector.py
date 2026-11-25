"""2차 목표: 문 주변 영역의 반려동물(고양이·개) 탐지.

COCO 사전학습 YOLO detect 모델을 그대로 쓴다. (클래스 15=cat, 16=dog)
비전 부하를 줄이기 위해 매 프레임이 아니라 N프레임마다 한 번 돌린다(vision.json의 pet.every_n).

지원 출력 형식
  A) YOLOv8/11 detect : (1, 4+80, N)  → NMS 필요
  B) end-to-end       : (1, K, 6)     행 = [x1, y1, x2, y2, conf, cls]
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..messages import PetDetection
from .door_geometry import DoorGeometry
from .onnx_backend import LetterboxInfo, OnnxModel, cxcywh_to_xyxy, letterbox, nms, to_blob, unletterbox_xy

COCO_PETS: Dict[int, str] = {15: "cat", 16: "dog"}


def decode_detect_output(out: np.ndarray, info: LetterboxInfo, classes: Dict[int, str],
                         conf_thr: float = 0.35, iou_thr: float = 0.5):
    a = np.asarray(out, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    if a.shape[1] == 6 and a.shape[0] != 6:                         # 형식 B
        cls = a[:, 5].astype(int)
        scores = a[:, 4]
        boxes = a[:, :4]
    else:                                                           # 형식 A (4+C, N) → (N, 4+C)
        a = a.T
        cls_scores = a[:, 4:]
        cls = cls_scores.argmax(1)
        scores = cls_scores.max(1)
        boxes = cxcywh_to_xyxy(a[:, :4])
    m = (scores >= conf_thr) & np.isin(cls, list(classes))
    boxes, scores, cls = boxes[m], scores[m], cls[m]
    keep = nms(boxes, scores, iou_thr)
    out_list = []
    for i in keep:
        bb = unletterbox_xy(boxes[i].reshape(2, 2), info).reshape(4)
        out_list.append((classes[int(cls[i])], float(scores[i]), bb))
    return out_list


class PetDetector:
    def __init__(self, model_path: str, input_size: Optional[int] = None, conf_thr: float = 0.35,
                 threads: int = 4, classes: Optional[Dict[int, str]] = None):
        self.model = OnnxModel(model_path, threads=threads)
        self.size = input_size or self.model.input_size
        self.conf_thr = conf_thr
        self.classes = classes or COCO_PETS

    def __call__(self, frame_bgr: np.ndarray, door: DoorGeometry) -> List[PetDetection]:
        img, info = letterbox(frame_bgr, self.size)
        out, _ = self.model(to_blob(img))
        res = []
        for label, score, bb in decode_detect_output(out, info, self.classes, self.conf_thr):
            res.append(PetDetection(label=label, score=round(score, 3),
                                    bbox=[round(float(v), 4) for v in bb],
                                    in_door_zone=door.bbox_in_door_zone(bb)))
        return res
