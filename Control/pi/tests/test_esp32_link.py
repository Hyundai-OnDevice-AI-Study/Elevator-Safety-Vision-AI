"""PROTOCOL.md 시험: Pi 쪽 Esp32Link 와 가짜 ESP32 를 시뮬레이션 시간으로 맞물려 돌린다.  (Control/pi 에서 python -m pytest -q)

시나리오 번호는 PROTOCOL.md 7장과 같다.
"""
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # Control/pi
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (_HERE, os.path.join(_ROOT, "Analysis")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from door_control.esp32_link import LINK_FAULT, NO_ACK_FAULT, Esp32Link  # noqa: E402
from door_control.fake_esp32 import FakeEsp32  # noqa: E402
from door_common.messages import ActionCommand, DeviceEvent  # noqa: E402


class Rig:
    """Pi(Esp32Link) ↔ 가짜 ESP32 를 10 ms 씩 진행. pi_quiet / esp_dead 로 한쪽 하트비트를 끊는다."""

    def __init__(self):
        self.t = 0.0
        self.link = Esp32Link(start_ts=0.0)
        self.esp = FakeEsp32(0.0)
        self.events = []          # (t, kind, value)
        self.esp_lines = []       # ESP32 가 보낸 줄 (dict, '#' 줄은 str)
        self.drop_to_esp = False  # Pi → ESP32 줄 버리기 (ack 안 옴 시험)
        self.esp_dead = False

    def _to_pi(self, lines):
        for ln in lines:
            s = ln.strip()
            self.esp_lines.append(s if s.startswith("#") else json.loads(s))
            for ev in self.link.feed_line(ln, self.t):
                self.events.append((self.t, ev.kind, ev.value))

    def _to_esp(self, lines):
        if self.drop_to_esp:
            return
        for ln in lines:
            self._to_pi(self.esp.feed_line(ln, self.t))

    def run(self, seconds):
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + 0.01, 6)
            if not self.esp_dead:
                self._to_pi(self.esp.step(self.t))
            lines, evs = self.link.tick(self.t)
            self.events += [(self.t, e.kind, e.value) for e in evs]
            self._to_esp(lines)

    def send(self, target, action):
        line = self.link.send(ActionCommand(self.t, target, action, {}), self.t)
        self._to_esp([line])

    def go(self, floor):
        self._to_pi(self.esp.press_go(floor, self.t))

    def ev(self, kind=None):
        return [(k, v) for _, k, v in self.events if kind is None or k == kind]

    def json_lines(self, key):
        return [d for d in self.esp_lines if isinstance(d, dict) and key in d]

    def clear(self):
        self.events.clear()
        self.esp_lines.clear()


def test_1_boot_reports_state_and_heartbeat():
    r = Rig()
    r.run(1.0)
    assert r.link.hello["proto"] == 1
    assert r.ev() == [("door", "closed"), ("car", "stopped")]
    assert 4 <= len(r.json_lines("hb")) <= 6                        # 200 ms 마다
    assert r.link.link_up


def test_2_trip_sequence():
    r = Rig()
    r.run(0.1)
    r.clear()
    r.go(3)
    r.run(6.0)
    assert r.ev() == [("car", "moving"), ("car", "stopped"), ("door", "open"), ("door", "closing"), ("door", "closed")]
    assert r.esp.floor == 3


def test_3_reopen_while_closing():
    r = Rig()
    r.run(0.1)
    r.go(2)
    r.run(1.2)
    r.go(2)                                                          # 같은 층 버튼 → 문 열림
    r.run(0.7)
    assert r.esp.door == "open"
    r.run(2.1)                                                       # 유지 시간 끝 → 닫히는 중
    assert r.esp.door == "closing"
    r.clear()
    r.send("door", "open")
    r.run(0.7)
    assert r.json_lines("ack")[0] == {"ack": r.link.seq}
    assert r.ev("door") == [("door", "open")]
    r.run(1.5)
    assert r.esp.door == "open"                                      # 유지 시간을 처음부터 다시 셈


def test_4_5_emergency_stop_latch_and_reset():
    r = Rig()
    r.run(0.1)
    r.go(4)
    r.run(1.5)
    assert r.esp.car == "moving"
    r.clear()
    t0 = r.t
    r.send("car", "emergency_stop")
    assert r.esp.car == "stopped"
    assert r.events[0][:3] == (t0, "car", "stopped")               # 같은 순간에 보고
    r.run(0.3)
    assert r.link.snapshot["es"] == 1
    r.go(2)
    r.run(1.0)
    assert r.esp.car == "stopped"                                    # 래치 중 버튼 무시
    r.send("car", "emergency_reset")
    r.run(0.3)
    assert r.link.snapshot["es"] == 0 and r.esp.car == "stopped"    # 저절로 안 움직임


def test_6_block_start_ignores_departure():
    r = Rig()
    r.run(0.1)
    r.send("car", "block_start")
    r.run(0.3)
    assert r.link.snapshot["blk"] == 1
    r.clear()
    r.go(3)
    r.run(2.0)
    assert ("car", "moving") not in r.ev()
    r.send("car", "block_end")
    r.go(3)
    r.run(0.2)
    assert ("car", "moving") in r.ev()


def test_7_door_open_refused_while_moving():
    r = Rig()
    r.run(0.1)
    r.go(3)
    r.run(0.5)
    r.send("door", "open")
    assert r.json_lines("ack")[-1] == {"ack": r.link.seq, "ok": 0, "err": "car_moving"}
    assert any("nack" in m and "car_moving" in m for _, m in r.link.log)


def test_8_duplicate_seq_executes_once():
    r = Rig()
    r.run(0.1)
    line = r.link.send(ActionCommand(r.t, "car", "block_start", {}), r.t)
    r._to_esp([line])
    r.esp.blk = False                                                # 실행됐는지 구분하려고 지움
    r._to_esp([line])                                                # 재전송
    assert r.esp.blk is False                                        # 다시 실행 안 함
    assert len(r.json_lines("ack")) == 2                             # ack 는 두 번


def test_9_pi_lost_blocks_departure_but_finishes_trip():
    r = Rig()
    r.run(0.1)
    r.link.hb_enabled = False
    r.run(0.8)
    assert not r.esp.pi_alive
    r.clear()
    r.go(2)
    r.run(2.0)
    assert ("car", "moving") not in r.ev()
    # 이동 중 끊김 → 이번 이동은 끝낸다
    r.link.hb_enabled = True
    r.run(0.3)
    r.go(3)
    r.run(1.2)
    assert r.esp.car == "moving"
    r.link.hb_enabled = False
    r.run(3.0)
    assert r.esp.floor == 3 and r.esp.car == "stopped"


def test_link_down_and_up_events():
    r = Rig()
    r.run(0.5)
    r.esp_dead = True
    r.run(1.0)
    assert ("fault", LINK_FAULT) in r.ev("fault")
    assert not r.link.link_up
    r.clear()
    r.esp_dead = False
    r.run(0.3)
    assert r.ev("fault") == [("fault", "clear")]


def test_no_ack_retries_then_fault():
    r = Rig()
    r.run(0.1)
    r.clear()
    r.drop_to_esp = True
    sent = []
    orig = r._to_esp
    r._to_esp = lambda lines: sent.extend(l for l in lines if "seq" in l) or orig(lines)
    r.send("car", "block_start")
    r.run(0.6)
    assert len(sent) == 4                                            # 처음 1 + 재전송 3
    assert ("fault", NO_ACK_FAULT) in r.ev("fault")


def test_heartbeat_resyncs_lost_event():
    link = Esp32Link(start_ts=0.0)
    link.feed_line('{"kind":"door","value":"closed"}', 0.1)
    evs = link.feed_line('{"hb":3,"door":"open","car":"stopped","blk":0,"es":0,"fl":1}', 0.2)   # door open 줄이 사라졌다고 가정
    assert [(e.kind, e.value) for e in evs] == [("door", "open"), ("car", "stopped")]


def test_ignores_debug_and_garbage():
    link = Esp32Link(start_ts=0.0)
    assert link.feed_line("# Moving Up", 0.1) == []
    assert link.feed_line("Floor Detected!", 0.1) == []
    assert link.feed_line('{"kind":"door","value":"ajar"}', 0.1) == []
    assert len(link.log) == 3


def test_non_esp32_targets_not_sent():
    link = Esp32Link()
    assert link.send(ActionCommand(0.0, "buzzer", "on", {"pattern": "short"}), 0.0) is None
    assert link.pending == {}


def test_with_state_machine_link_loss_gives_fault_and_block():
    """연결이 끊기면 상태 머신이 FAULT → 출발 보류 명령을 낸다 (ESP32 로는 못 가지만 Pi 쪽 기록은 남음)."""
    from door_analysis.state import FAULT, SafetyStateMachine
    sm = SafetyStateMachine({}, door="closed", car="stopped", start_ts=0.0)
    link = Esp32Link(start_ts=0.0)
    out = []
    t = 0.0
    while t < 1.0:
        t = round(t + 0.05, 6)
        out += sm.on_pose(t, 0, [])                                  # 비전은 정상
        _, evs = link.tick(t)
        for e in evs:
            out += sm.handle(e)
    assert sm.state == FAULT
    assert any(isinstance(m, ActionCommand) and m.action == "block_start" for m in out)


def test_bridge_with_fake_transport_realtime():
    """실제 스레드·시간으로: 가짜 ESP32 이동 중 비상정지."""
    from door_control.esp32_bridge import Esp32Bridge
    br = Esp32Bridge(fake=True)
    evs = []
    try:
        t_end = time.monotonic() + 0.3
        while time.monotonic() < t_end:
            evs += br.poll()
            time.sleep(0.02)
        br.io.press_go(3)
        t_end = time.monotonic() + 0.3
        while time.monotonic() < t_end:
            evs += br.poll()
            time.sleep(0.02)
        assert br.send(ActionCommand(0.0, "car", "emergency_stop", {}))
        t_end = time.monotonic() + 0.4
        while time.monotonic() < t_end:
            evs += br.poll()
            time.sleep(0.02)
    finally:
        br.close()
    kv = [(e.kind, e.value) for e in evs if isinstance(e, DeviceEvent)]
    assert kv[:2] == [("door", "closed"), ("car", "stopped")]
    assert ("car", "moving") in kv and kv[-1] == ("car", "stopped")
    assert br.link.snapshot.get("es") == 1
    assert br.link.pending == {}
