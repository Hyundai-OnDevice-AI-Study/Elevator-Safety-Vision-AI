"""공유 코드 테스트: 메시지 형식, 전송, 문 기하.  (common/ 에서 python -m pytest -q)"""
import multiprocessing as mp
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("common",):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))

import pytest  # noqa: E402,F401

from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_common.fake_pose import SIM_DOOR_CONFIG, generate  # noqa: E402
from door_common.messages import (ActionCommand, DeviceEvent, PressureSample,  # noqa: E402
                                  StateChange, from_json, to_json)
from door_common.transport import QueuePublisher, QueueSubscriber  # noqa: E402

DOOR = DoorGeometry.from_config(SIM_DOOR_CONFIG)



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


def test_door_signed_distance():
    assert DOOR.signed_distance((0.5, 0.5)) > 0           # 카 안쪽
    assert DOOR.signed_distance((0.9, 0.5)) < 0           # 문 바깥
    assert DOOR.signed_distance((0.72, 0.3)) == pytest.approx(0, abs=1e-6)
    assert DOOR.in_button_panel((0.6, 0.4))


def test_polygons_exclude_and_ignore():
    from door_common.door_geometry import point_in_polygon
    sq = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    assert point_in_polygon((0.2, 0.2), sq) and not point_in_polygon((0.4, 0.2), sq)
    cfg = {"door": {**SIM_DOOR_CONFIG["door"], "exclude": [sq], "ignore": [[(0.0, 0.8), (0.3, 0.8), (0.3, 1.0), (0.0, 1.0)]]}}
    d = DoorGeometry.from_config(cfg)
    assert d.in_button_panel((0.2, 0.2))                       # 다각형 패널
    assert d.in_button_panel((0.60, 0.40))                     # 기존 사각형 패널도 그대로
    kp = [[0.5, 0.5, 0.9]] * 15 + [[0.15, 0.9, 0.9], [0.2, 0.9, 0.9]]   # 발목이 무시 영역 안
    assert d.in_ignore(kp)
    kp2 = [[0.5, 0.5, 0.9]] * 15 + [[0.5, 0.9, 0.9], [0.55, 0.9, 0.9]]
    assert not d.in_ignore(kp2)
    s = d.scaled(1.78)                                          # 화면비 보정해도 같은 판정
    assert s.in_button_panel((0.2 * 1.78, 0.2))
