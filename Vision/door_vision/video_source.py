"""영상 파일을 '카메라처럼' 재생하는 소스.

중요: 실제 카메라는 추론이 느려도 기다려 주지 않는다. 처리하는 동안 지나간 프레임은 버려진다.
realtime=True 이면 벽시계 기준으로 지금 보여야 할 프레임으로 건너뛰므로,
Pi에서 FPS가 낮을 때 판단이 실제로 얼마나 늦어지는지 정직하게 재현된다.
realtime=False 이면 모든 프레임을 순서대로 처리한다 (정확도 평가·디버깅용).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoFileSource:
    def __init__(self, path: str, realtime: bool = True, loop: bool = False,
                 resize_width: Optional[int] = None):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.realtime, self.loop, self.resize_width = realtime, loop, resize_width
        self.idx = -1
        self.t_start: Optional[float] = None
        self.dropped = 0

    def _restart(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.idx = -1
        self.t_start = time.monotonic()

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """(성공 여부, 프레임, 영상 내 프레임 번호)."""
        if self.t_start is None:
            self.t_start = time.monotonic()
        if self.realtime:
            target = int((time.monotonic() - self.t_start) * self.fps)
            if target <= self.idx:                       # 너무 빨리 왔으면 다음 프레임 시각까지 대기
                time.sleep(max(0.0, (self.idx + 1) / self.fps - (time.monotonic() - self.t_start)))
                target = self.idx + 1
            while self.idx < target - 1:                 # 밀린 프레임은 디코딩 없이 건너뜀
                if not self.cap.grab():
                    break
                self.idx += 1
                self.dropped += 1
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self._restart()
                ok, frame = self.cap.read()
            if not ok:
                return False, None, self.idx
        self.idx += 1
        if self.resize_width and frame.shape[1] != self.resize_width:
            h = int(frame.shape[0] * self.resize_width / frame.shape[1])
            frame = cv2.resize(frame, (self.resize_width, h))
        return True, frame, self.idx

    def release(self) -> None:
        self.cap.release()
