"""pytest -q  로 실행. 모델 파일 없이 돌아가는 테스트만 모았다."""
import multiprocessing as mp
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from door_drag.analysis.pose_features import PoseAnalyzer  # noqa: E402
from door_drag.messages import (ActionCommand, DeviceEvent, PoseFrame, PressureSample,  # noqa: E402
                                StateChange, from_json, to_json)
from door_drag.sim.fake_pose import EXPECTED, SCENARIOS, SIM_DOOR_CONFIG, generate  # noqa: E402
from door_drag.transport import QueuePublisher, QueueSubscriber  # noqa: E402
from door_drag.vision.door_geometry import DoorGeometry  # noqa: E402
from door_drag.vision.onnx_backend import letterbox, nms  # noqa: E402
from door_drag.vision.pet_detector import COCO_PETS, decode_detect_output  # noqa: E402
from door_drag.vision.pose_estimator import decode_pose_output  # noqa: E402
from door_drag.vision.tracker import IouTracker  # noqa: E402

DOOR = DoorGeometry.from_config(SIM_DOOR_CONFIG)


# ------------------------------------------------------------- 메시지
def test_messages_roundtrip():
    pf = next(generate("normal"))
    msgs = [pf, PressureSample(1.0, 123, 4.5), DeviceEvent(1.0, "door", "closed"),
            StateChange(1.0, "MONITORING", "DANGER_PRE", "pose", {"level": 3}),
            ActionCommand(1.0, "door", "open")]
    for m in msgs:
        back = from_json(to_json(m))
        assert type(back) is type(m)
        assert to_json(back) == to_json(m)


def test_queue_transport_between_processes():
    q = mp.Queue(maxsize=60)
    pub, sub = QueuePublisher(q), QueueSubscriber(q)
    for pf in generate("normal", duration=1.0):      # 15개
        pub.publish(pf)
    got = []
    for _ in range(20):
        got += sub.poll(timeout=0.05)
        if len(got) == 15:
            break
    assert [m.frame_id for m in got] == list(range(15))   # 순서·누락 없이 도착


def test_queue_overflow_never_blocks():
    q = mp.Queue(maxsize=5)
    pub = QueuePublisher(q)
    for pf in generate("normal", duration=3.0):      # 45개를 크기 5 큐에 → 막히지 않고 오래된 것부터 버림
        pub.publish(pf)
    assert pub.dropped >= 0


# ------------------------------------------------------------- 전처리·후처리
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


# ------------------------------------------------------------- 문 기하
def test_door_signed_distance():
    assert DOOR.signed_distance((0.5, 0.5)) > 0           # 카 안쪽
    assert DOOR.signed_distance((0.9, 0.5)) < 0           # 문 바깥
    assert DOOR.signed_distance((0.72, 0.3)) == pytest.approx(0, abs=1e-6)
    assert DOOR.in_button_panel((0.6, 0.4))


# ------------------------------------------------------------- 자세 규칙
@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("fps", [8, 15, 30])
@pytest.mark.parametrize("seed", [0, 1])
def test_scenarios(name, fps, seed):
    a = PoseAnalyzer(DOOR)
    levels, flags = [], set()
    for pf in generate(name, fps=fps, seed=seed, noise=0.004):
        r = a.update(pf)
        levels.append((pf.ts, r.level))
        for p in r.persons:
            flags |= {k for k, v in p.flags.items() if v}
    exp = EXPECTED[name]
    assert max(l for _, l in levels) == exp["max_level"]
    if "danger_after" in exp:      # 사건 전에는 DANGER 가 나오면 안 된다
        assert all(l < 3 for t, l in levels if t < exp["danger_after"])
    if "flag" in exp:
        assert exp["flag"] in flags


def test_reset_on_door_open():
    a = PoseAnalyzer(DOOR)
    for pf in generate("dragged"):
        a.update(pf)
    a.reset()
    assert a.tracks == {}


def test_pet_dwell():
    a = PoseAnalyzer(DOOR)
    res = [a.update(pf, {"dwell_s": 0.5}).pet_near_door for pf in generate("normal", with_pet=True)]
    assert not res[0] and res[-1]


def test_aspect_ratio_invariance():
    """같은 장면을 16:9 영상의 정규화 좌표로 보내도 판단이 같아야 한다."""
    import copy
    aspect = 1280 / 720
    d = copy.deepcopy(SIM_DOOR_CONFIG)
    for k in ("gap_p1", "gap_p2", "inside_point"):
        d["door"][k][0] /= aspect
    b = d["door"]["button_panel"]
    d["door"]["button_panel"] = [b[0] / aspect, b[1], b[2] / aspect, b[3]]
    d["door"]["zone_width"] /= aspect
    wide = DoorGeometry.from_config(d)
    for name in ("dragged", "walk_out", "brace", "leash_yank"):
        a1, a2 = PoseAnalyzer(DOOR), PoseAnalyzer(wide)
        l1, l2 = [], []
        for pf in generate(name, noise=0.003):
            l1.append(a1.update(pf).level)
            pf2 = copy.deepcopy(pf)
            pf2.img_w, pf2.img_h = 1280, 720
            for p in pf2.persons:
                for kp in p.keypoints:
                    kp[0] /= aspect
            l2.append(a2.update(pf2).level)
        assert max(l1) == max(l2), name


def test_real_cam03_leash_drag():
    """실제 영상(YOLO11n-pose 320)에서 뽑은 관절 데이터로 회귀 테스트.
    규칙을 바꿨을 때 이 영상의 결과가 나빠지면 바로 알 수 있다."""
    import json
    from door_drag.messages import PoseFrame as PF
    from door_drag.transport import read_jsonl
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(os.path.join(here, "..", "config", "vision_cam03.json"), encoding="utf-8"))
    frames = [f for f in read_jsonl(os.path.join(here, "data", "cam03_leash_drag.jsonl")) if isinstance(f, PF)]
    a = PoseAnalyzer(DoorGeometry.from_config(cfg))
    t0 = frames[0].ts
    levels = [(f.ts - t0, a.update(f).level) for f in frames]
    assert all(l == 0 for t, l in levels if t < 3.9)                 # 사건 전(가만히 서서 휴대폰)에는 정상
    assert any(l >= 2 for t, l in levels if t < 5.5)                 # 목줄에 팔이 당겨지면 경계
    assert any(l == 3 for t, l in levels)                            # 끌려가면 위험
