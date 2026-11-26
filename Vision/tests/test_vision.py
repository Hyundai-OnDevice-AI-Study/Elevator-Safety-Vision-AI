"""비전 테스트: 전처리·후처리·추적. 모델 파일 없이 돈다.  (Vision/ 에서 python -m pytest -q)"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("Vision",):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))

import numpy as np  # noqa: E402
import pytest  # noqa: E402,F401

from door_vision.onnx_backend import letterbox, nms  # noqa: E402
from door_vision.pet_detector import COCO_PETS, decode_detect_output  # noqa: E402
from door_vision.pose_estimator import decode_pose_output  # noqa: E402
from door_vision.tracker import IouTracker  # noqa: E402


def test_letterbox_keeps_ratio():
    img = np.zeros((480, 640, 3), np.uint8)
    out, info = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert info.scale == pytest.approx(0.5) and info.pad_y == 40 and info.pad_x == 0


def _letterbox_info(w=640, h=480, size=320):
    return letterbox(np.zeros((h, w, 3), np.uint8), size)[1]


def test_decode_pose_format_A_with_nms():
    info = _letterbox_info()
    n = 100
    out = np.zeros((1, 56, n), np.float32)
    # 같은 사람 박스 2개(겹침) + 다른 사람 1개 + 낮은 점수 다수
    for j, (cx, cy, s) in enumerate([(100, 160, 0.9), (102, 161, 0.8), (250, 160, 0.7)]):
        out[0, :5, j] = [cx, cy, 60, 200, s]
        out[0, 5::3, j] = cx          # 키포인트 x
        out[0, 6::3, j] = cy          # 키포인트 y
        out[0, 7::3, j] = 0.9
    out[0, 4, 3:] = 0.1
    res = decode_pose_output(out, info, conf_thr=0.4)
    assert len(res) == 2
    assert res[0].score == pytest.approx(0.9)
    # 입력 (100, 160) → 원본 (200, 240)/(640,480) = (0.3125, 0.5)
    assert res[0].kpts[0, 0] == pytest.approx(0.3125, abs=1e-3)
    assert res[0].kpts[0, 1] == pytest.approx(0.5, abs=1e-3)


def test_decode_pose_format_B_end2end():
    info = _letterbox_info()
    out = np.zeros((1, 300, 57), np.float32)
    out[0, 0, :6] = [70, 60, 130, 260, 0.85, 0]
    out[0, 0, 6::3], out[0, 0, 7::3], out[0, 0, 8::3] = 100, 160, 0.9
    res = decode_pose_output(out, info)
    assert len(res) == 1 and res[0].kpts[5, 0] == pytest.approx(0.3125, abs=1e-3)


def test_decode_detect_pets_only():
    info = _letterbox_info()
    out = np.zeros((1, 84, 10), np.float32)
    out[0, :4, 0] = [100, 200, 40, 40]; out[0, 4 + 16, 0] = 0.8      # dog
    out[0, :4, 1] = [200, 200, 40, 40]; out[0, 4 + 0, 1] = 0.9       # person → 무시
    res = decode_detect_output(out, info, COCO_PETS)
    assert [r[0] for r in res] == ["dog"]


def test_nms():
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    assert sorted(nms(b, np.array([0.9, 0.8, 0.7]), 0.5).tolist()) == [0, 2]


def test_tracker_keeps_id():
    t = IouTracker()
    a = t.update([np.array([0.1, 0.1, 0.3, 0.6])])
    b = t.update([np.array([0.5, 0.1, 0.7, 0.6]), np.array([0.11, 0.1, 0.31, 0.6])])
    assert b[1] == a[0] and b[0] != a[0]
