"""ONNX 모델 실행기 + 전처리(letterbox).

우선 onnxruntime(CPU)을 쓰고, 없으면 OpenCV DNN으로 대체한다.
Raspberry Pi 4B(ARM Cortex-A72 4코어)에서는 onnxruntime이 보통 더 빠르다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class LetterboxInfo:
    scale: float      # 원본 → 입력 크기 배율
    pad_x: float      # 좌우 패딩(px, 입력 이미지 기준)
    pad_y: float      # 상하 패딩
    src_w: int
    src_h: int


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)) -> Tuple[np.ndarray, LetterboxInfo]:
    """비율을 유지하며 size x size 로 맞추고 남는 곳은 회색으로 채운다.

    YOLO는 학습 때 이 방식으로 입력을 만들었기 때문에 추론 때도 똑같이 해야
    정확도가 떨어지지 않는다. (그냥 resize 하면 사람이 찌그러져 자세가 틀어진다)
    """
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, LetterboxInfo(scale, left, top, w, h)


def to_blob(img_bgr: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 → RGB NCHW float32 [0,1]."""
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class OnnxModel:
    def __init__(self, path: str, threads: int = 4, backend: str = "auto"):
        self.path = path
        self.backend = None
        if backend in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
                inp = self.sess.get_inputs()[0]
                self.input_name = inp.name
                self.input_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
                self.backend = "onnxruntime"
            except ImportError:
                if backend == "onnxruntime":
                    raise
        if self.backend is None:
            self.net = cv2.dnn.readNetFromONNX(path)
            cv2.setNumThreads(threads)
            self.input_size = 640  # OpenCV는 입력 크기를 알려주지 않으므로 설정값으로 덮어쓴다
            self.backend = "opencv"

    def __call__(self, blob: np.ndarray) -> Tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        if self.backend == "onnxruntime":
            out = self.sess.run(None, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        return out, (time.perf_counter() - t0) * 1000.0


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """순수 numpy NMS. 겹치는 박스 중 점수가 가장 높은 것만 남긴다."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def unletterbox_xy(xy: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """입력(letterbox) 좌표 → 원본 이미지 기준 0~1 정규화 좌표."""
    out = xy.astype(np.float32).copy()
    out[..., 0] = (out[..., 0] - info.pad_x) / info.scale / info.src_w
    out[..., 1] = (out[..., 1] - info.pad_y) / info.scale / info.src_h
    return np.clip(out, 0.0, 1.0)
