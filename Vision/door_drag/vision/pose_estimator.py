"""YOLO-pose ONNX 추론 + 후처리.

지원하는 출력 형식 (Ultralytics export 기준)
  A) YOLOv8/YOLO11-pose : (1, 56, N)   행 = [cx, cy, w, h, conf, 17 x (x, y, v)]  → NMS 필요
  B) end-to-end(YOLO26 등): (1, K, 57)  행 = [x1, y1, x2, y2, conf, cls, 17 x (x, y, v)] → NMS 불필요
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .keypoints import NUM_KPTS
from .onnx_backend import LetterboxInfo, OnnxModel, cxcywh_to_xyxy, letterbox, nms, to_blob, unletterbox_xy


@dataclass
class RawPose:
    bbox: np.ndarray        # (4,) 정규화 xyxy
    score: float
    kpts: np.ndarray        # (17, 3) 정규화 x, y, conf


def decode_pose_output(out: np.ndarray, info: LetterboxInfo, conf_thr: float = 0.4,
                       iou_thr: float = 0.5, max_det: int = 5) -> List[RawPose]:
    """모델 원시 출력 → RawPose 리스트. (모델 없이도 테스트할 수 있게 분리)"""
    a = np.asarray(out, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]
    kdim = NUM_KPTS * 3

    if a.shape[0] == 5 + kdim and a.shape[1] != 5 + kdim:        # 형식 A: (56, N)
        a = a.T
        scores = a[:, 4]
        m = scores >= conf_thr
        a, scores = a[m], scores[m]
        boxes = cxcywh_to_xyxy(a[:, :4])
        keep = nms(boxes, scores, iou_thr)[:max_det]
        boxes, scores, kp = boxes[keep], scores[keep], a[keep, 5:5 + kdim]
    elif a.shape[1] == 6 + kdim:                                  # 형식 B: (K, 57)
        scores = a[:, 4]
        m = scores >= conf_thr
        a, scores = a[m], scores[m]
        order = scores.argsort()[::-1][:max_det]
        boxes, scores, kp = a[order, :4], scores[order], a[order, 6:6 + kdim]
    else:
        raise ValueError(f"지원하지 않는 pose 출력 shape: {out.shape}")

    results = []
    for b, s, k in zip(boxes, scores, kp):
        k = k.reshape(NUM_KPTS, 3).copy()
        k[:, :2] = unletterbox_xy(k[:, :2], info)
        # 일부 export는 visibility를 logit으로 내보낸다 → 0~1로 변환
        if k[:, 2].min() < 0 or k[:, 2].max() > 1:
            k[:, 2] = 1 / (1 + np.exp(-k[:, 2]))
        bb = unletterbox_xy(b.reshape(2, 2), info).reshape(4)
        results.append(RawPose(bbox=bb, score=float(s), kpts=k))
    return results


class PoseEstimator:
    def __init__(self, model_path: str, input_size: Optional[int] = None, conf_thr: float = 0.4,
                 iou_thr: float = 0.5, max_det: int = 5, threads: int = 4, backend: str = "auto"):
        self.model = OnnxModel(model_path, threads=threads, backend=backend)
        self.size = input_size or self.model.input_size
        self.conf_thr, self.iou_thr, self.max_det = conf_thr, iou_thr, max_det
        self.last_timing = {}

    def __call__(self, frame_bgr: np.ndarray) -> List[RawPose]:
        import time
        t0 = time.perf_counter()
        img, info = letterbox(frame_bgr, self.size)
        blob = to_blob(img)
        t1 = time.perf_counter()
        out, infer_ms = self.model(blob)
        t2 = time.perf_counter()
        poses = decode_pose_output(out, info, self.conf_thr, self.iou_thr, self.max_det)
        t3 = time.perf_counter()
        self.last_timing = {"pre_ms": (t1 - t0) * 1e3, "infer_ms": infer_ms, "post_ms": (t3 - t2) * 1e3}
        return poses
