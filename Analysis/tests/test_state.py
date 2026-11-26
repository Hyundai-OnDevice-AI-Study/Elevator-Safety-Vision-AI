"""상태 머신 테스트: 메시지를 순서대로 넣고 상태·명령을 확인한다.  (Analysis/ 에서 python -m pytest -q)"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _d in ("Analysis",):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))

from door_analysis.state import (DANGER_MOVING, DANGER_PRE, DOOR_OPEN, FAULT, MONITORING,  # noqa: E402
                                 WARNING, SafetyStateMachine)
from door_common.messages import ActionCommand, DeviceEvent, PressureSample, StateChange  # noqa: E402


def cmds(outs):
    return [(m.target, m.action) for m in outs if isinstance(m, ActionCommand)]


def states(outs):
    return [m.state for m in outs if isinstance(m, StateChange)]


class Sim:
    """시각을 0.1초씩 올리며 자세를 넣는 도우미. 비전 타임아웃이 안 나도록 매 단계 자세를 넣는다."""

    def __init__(self, rules=None, door="closed", car="stopped"):
        self.sm = SafetyStateMachine(rules or {}, door=door, car=car, start_ts=0.0)
        self.t = 0.0
        self.log = []

    def pose(self, level=0, seconds=0.1, **kw):
        out = []
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + 0.1, 3)
            out += self.sm.on_pose(self.t, level, [], **kw)
        self.log += out
        return out

    def dev(self, kind, value):
        out = self.sm.on_device(DeviceEvent(self.t, kind, value))
        self.log += out
        return out

    def press(self, grams, seconds=0.1, level=0):
        out = []
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + 0.1, 3)
            out += self.sm.on_pressure(PressureSample(self.t, 0, grams))
            out += self.sm.on_pose(self.t, level, [])
        self.log += out
        return out


def test_door_cycle():
    s = Sim(door="open")
    assert s.sm.state == DOOR_OPEN
    s.pose(0)
    out = s.dev("door", "closing")
    assert states(out) == [MONITORING] and ("buzzer", "off") in cmds(out)
    assert states(s.dev("door", "open")) == [DOOR_OPEN]


def test_warning_blocks_start_and_clears_after_hold():
    s = Sim()
    s.pose(0, 0.5)
    out = s.pose(2)
    assert states(out) == [WARNING]
    assert ("car", "block_start") in cmds(out) and ("beacon", "yellow") in cmds(out)
    out = s.pose(0, 1.4)
    assert s.sm.state == WARNING                       # 경고 유지 시간 1.5초
    out = s.pose(0, 0.3)
    assert states(out) == [MONITORING] and ("car", "block_end") in cmds(out)


def test_leaning_warns_without_blocking():
    s = Sim()
    out = s.pose(0, leaning=True)
    assert states(out) == [WARNING]
    assert ("car", "block_start") not in cmds(out)
    say = [m.params["text"] for m in out if isinstance(m, ActionCommand) and m.target == "voice"]
    assert say and "기대" in say[0]


def test_danger_while_stopped_reopens_door_and_latches_until_open():
    s = Sim()
    out = s.pose(3)
    assert states(out) == [DANGER_PRE]
    c = cmds(out)
    assert ("door", "open") in c and ("car", "block_start") in c and ("notify", "emergency") in c
    s.pose(0, 2.0)
    assert s.sm.state == DANGER_PRE                    # 해석 수준이 내려가도 문이 열릴 때까지 유지
    out = s.dev("door", "open")
    assert states(out) == [DOOR_OPEN] and ("car", "block_end") in cmds(out)


def test_danger_pre_becomes_moving_if_car_moves():
    s = Sim()
    s.pose(3)
    out = s.dev("car", "moving")
    assert states(out) == [DANGER_MOVING] and ("car", "emergency_stop") in cmds(out)
    out += s.dev("car", "stopped") + s.pose(0, 1.0)
    assert ("car", "block_end") not in cmds(out)       # 비상정지 중에는 출발 보류를 풀지 않음


def test_danger_while_moving_needs_manual_reset():
    s = Sim(car="moving")
    out = s.pose(3)
    assert states(out) == [DANGER_MOVING]
    assert ("car", "emergency_stop") in cmds(out) and ("door", "open") not in cmds(out)
    s.dev("car", "stopped")
    s.pose(0, 1.0)
    s.dev("door", "open")
    assert s.sm.state == DANGER_MOVING                 # 문이 열려도 관리자 리셋 전까지 유지
    out = s.dev("operator", "reset")
    assert ("car", "emergency_reset") in cmds(out) and states(out) == [DOOR_OPEN]


def test_danger_moving_reset_on_door_open_option():
    s = Sim({"state": {"danger_moving_reset": "door_open"}}, car="moving")
    s.pose(3)
    s.dev("car", "stopped")
    out = s.dev("door", "open")
    assert states(out) == [DOOR_OPEN] and ("car", "emergency_reset") in cmds(out)


def test_reset_while_danger_remains_goes_back_to_danger_pre():
    s = Sim(car="moving")
    s.pose(3)
    s.dev("car", "stopped")
    out = s.dev("operator", "reset")
    out += s.pose(3)
    assert s.sm.state == DANGER_PRE and ("door", "open") in cmds(out)


def test_vision_timeout_is_fault_and_recovers():
    sm = SafetyStateMachine({}, door="closed", car="stopped", start_ts=0.0)
    sm.on_pose(0.1, 0, [])
    out = sm.tick(1.5)                                  # 1.4초 동안 PoseFrame 없음
    assert states(out) == [FAULT] and ("car", "block_start") in cmds(out) and ("notify", "fault") in cmds(out)
    out = sm.on_pose(1.6, 0, [])
    assert states(out) == [MONITORING] and ("car", "block_end") in cmds(out)


def test_vision_error_status_is_fault():
    s = Sim()
    s.pose(0)
    out = s.pose(0, 1.3, status="error")
    assert FAULT in states(out)


def test_danger_overrides_fault():
    s = Sim()
    s.pose(0)
    s.dev("fault", "door_sensor")
    assert s.sm.state == FAULT
    assert states(s.pose(3)) == [DANGER_PRE]
    s.dev("fault", "clear")


def test_pressure_rules():
    rules = {"pressure": {"threshold_g": 50, "sustain_s": 0.3}}
    s = Sim(rules)
    s.press(1000, 0.3)                                  # 첫 값이 영점
    s.press(1030, 1.0)
    assert s.sm.state == MONITORING                     # 기준 미만
    s.press(1100, 0.2)
    assert s.sm.state == MONITORING                     # 순간 충격(0.3초 미만)은 무시
    s.press(1000, 0.5)
    s.press(1100, 0.5)
    assert s.sm.state == WARNING                        # 압력만으로는 경고
    out = s.press(1100, 0.1, level=2)
    assert states(out) == [DANGER_PRE]                  # 버팀 자세 + 압력 → 위험


def test_pet_near_door_with_pressure_is_danger():
    rules = {"pressure": {"threshold_g": 50, "sustain_s": 0.2}}
    sm = SafetyStateMachine(rules, door="closed", car="stopped", start_ts=0.0)
    sm.on_pressure(PressureSample(0.1, 0, 0))
    sm.on_pose(0.1, 0, [], pet_near=True)
    assert sm.state == WARNING
    for i in range(2, 6):
        sm.on_pressure(PressureSample(i / 10, 0, 80))
        sm.on_pose(i / 10, 0, [], pet_near=True)
    assert sm.state == DANGER_PRE


def test_pressure_baseline_resets_on_door_open():
    rules = {"pressure": {"threshold_g": 50, "sustain_s": 0.0}}
    sm = SafetyStateMachine(rules, door="closed", car="stopped", start_ts=0.0)
    sm.on_pressure(PressureSample(0.1, 0, 0))
    sm.on_device(DeviceEvent(0.2, "door", "open"))
    sm.on_pressure(PressureSample(0.3, 0, 500))          # 문 열린 뒤 첫 값 = 새 영점 (예: 사람이 기댄 채로 영점)
    sm.on_device(DeviceEvent(0.4, "door", "closed"))
    sm.on_pose(0.5, 0, [])
    sm.on_pressure(PressureSample(0.5, 0, 520))
    assert sm.state == MONITORING


def test_commands_only_on_transition():
    s = Sim()
    first = s.pose(2)
    again = s.pose(2, 1.0)
    assert cmds(first) and not cmds(again)              # 같은 상태에서 명령을 반복하지 않음


def test_pipeline_resets_pose_on_door_open_and_runs_cam03():
    """해석 + 상태 머신: cam03 재생 → 카 이동 중 위험 → DANGER_MOVING."""
    import json
    sys.path.insert(0, os.path.join(_ROOT, "Analysis", "tools"))
    from run_state import run
    from door_common.door_geometry import DoorGeometry
    from door_common.messages import PoseFrame
    from door_common.transport import read_jsonl
    cfg = json.load(open(os.path.join(_ROOT, "Vision", "config", "vision_cam03.json"), encoding="utf-8"))
    rules = json.load(open(os.path.join(_ROOT, "Analysis", "config", "rules.json"), encoding="utf-8"))
    frames = [f for f in read_jsonl(os.path.join(_ROOT, "Analysis", "tests", "data", "cam03_leash_drag.jsonl"))
              if isinstance(f, PoseFrame)]
    events = [[0.0, "door", "open"], [0.8, "door", "closing"], [1.5, "door", "closed"], [2.0, "car", "moving"]]
    final, log = run(frames, DoorGeometry.from_config(cfg), rules, events, quiet=True)
    seq = [(round(t, 2), m.state) for t, m in log if isinstance(m, StateChange)]
    assert [s for _, s in seq] == [MONITORING, WARNING, DANGER_MOVING]
    assert next(t for t, s in seq if s == DANGER_MOVING) <= 6.2
    assert final == DANGER_MOVING
