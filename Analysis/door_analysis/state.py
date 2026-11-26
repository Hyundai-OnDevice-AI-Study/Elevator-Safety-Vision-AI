"""안전 상태 머신 — 자세 해석 · 압력 · 문/카 상태를 합쳐 상태를 정하고, 제어 명령을 낸다.

입출력이 없는 순수 클래스다. 메시지를 넣으면 '나갈 메시지 목록'(StateChange, ActionCommand)이 나온다.
그래서 영상 재생 · 실시간 · 테스트에서 같은 코드를 쓴다. 시간은 메시지의 ts(monotonic 초)만 쓴다.

상태 (common/door_common/messages.py StateChange)
  DOOR_OPEN      문 열림. 끼일 수 없음 → 판정 초기화, 경보 끔
  MONITORING     문 닫힘/닫히는 중, 근거 없음 → 정상 운행
  WARNING        주의 근거 (버팀 자세, 압력 상승, 문에 기댐, 반려동물이 문 앞) → 경고, 필요하면 출발 보류
  DANGER_PRE     카가 서 있을 때 위험 근거 → 문을 다시 열어 줄을 풀어 줌 + 출발 보류
  DANGER_MOVING  카가 움직일 때 위험 근거 → 비상정지 (문은 열 수 없음)
  FAULT          비전이 안 들어옴 · 장치 고장 → 출발 보류 (고장 난 채로 운행하지 않음)

우선순위: DANGER_MOVING > DANGER_PRE > FAULT > WARNING > MONITORING

래치
  DANGER_PRE    : 문이 열릴 때까지 유지 (해석 수준이 내려가도)
  DANGER_MOVING : 관리자 리셋(DeviceEvent kind="operator", value="reset")까지 유지.
                  rules.json state.danger_moving_reset = "door_open" 이면 문이 열릴 때 해제

출발 보류(car block_start / block_end)는 상태와 따로 관리한다.
  WARNING(일부 근거)·FAULT·DANGER_PRE 에서 카가 서 있으면 block_start, MONITORING·DOOR_OPEN 으로 돌아오면 block_end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from door_common.messages import ActionCommand, DeviceEvent, PressureSample, StateChange

DOOR_OPEN, MONITORING, WARNING = "DOOR_OPEN", "MONITORING", "WARNING"
DANGER_PRE, DANGER_MOVING, FAULT = "DANGER_PRE", "DANGER_MOVING", "FAULT"
STATES = (DOOR_OPEN, MONITORING, WARNING, DANGER_PRE, DANGER_MOVING, FAULT)

DEFAULT_STATE_RULES = {
    "danger_moving_reset": "manual",     # "manual": 관리자 리셋 / "door_open": 문이 열리면 해제
    "warning_hold_s": 1.5,               # 경고 근거가 사라져도 이만큼은 WARNING 유지 (깜빡임 방지)
    "danger_pose_level": 3,              # 자세 수준이 이 이상이면 위험 근거
    "warning_pose_level": 2,             # 자세 수준이 이 이상이면 경고 근거 (압력과 겹치면 위험)
    "warn_on_leaning": True,             # 문에 기댄 사람 → 경고 ("기대기 → 경고" 시나리오)
    "warn_on_pet_near_door": True,       # 반려동물이 문 앞에 머묾 → 경고 (2차)
    "block_on_warning": ["pose", "pressure", "pet"],   # 이 근거로 WARNING 이면 출발 보류 (기댐만으로는 보류 안 함)
    "notify_on_danger": True,            # 위험이면 비상 전송 (1차: 출력만)
}
DEFAULT_PRESSURE_RULES = {
    "threshold_g": None,                 # 영점 대비 이 이상 눌리면 '압력 상승'. None 이면 압력은 쓰지 않음
    "sustain_s": 0.3,                    # 그 상태가 이만큼 계속돼야 인정 (순간 충격 무시)
    "timeout_s": None,                   # 압력 값이 이만큼 안 오면 고장. None 이면 검사 안 함
}

VOICE = {
    "lean": "문에 기대지 마세요.",
    "step_back": "문에서 떨어져 주세요.",
    "pet": "반려동물을 문에서 떨어뜨려 주세요.",
    "danger_pre": "위험합니다. 문을 엽니다.",
    "danger_moving": "비상 정지합니다. 움직이지 마세요.",
    "fault": "안전 장치 점검 중입니다. 잠시 기다려 주세요.",
}


@dataclass
class _Inputs:
    """상태 머신이 기억하는 최신 입력."""
    door: str = "open"                   # open / closing / closed
    car: str = "stopped"                 # stopped / moving
    pose_level: int = 0
    pose_reasons: List[str] = field(default_factory=list)
    leaning: bool = False
    pet_near: bool = False
    last_pose_ts: Optional[float] = None
    vision_bad_since: Optional[float] = None   # status 가 ok 가 아니게 된 시각
    p_baseline: Optional[float] = None
    p_excess: float = 0.0
    p_over_since: Optional[float] = None
    last_pressure_ts: Optional[float] = None
    device_faults: Set[str] = field(default_factory=set)


class SafetyStateMachine:
    def __init__(self, rules: Optional[dict] = None, door: str = "open", car: str = "stopped",
                 start_ts: float = 0.0):
        rules = rules or {}
        self.r = {**DEFAULT_STATE_RULES, **{k: v for k, v in rules.get("state", {}).items() if not k.startswith("_")}}
        self.p = {**DEFAULT_PRESSURE_RULES, **{k: v for k, v in rules.get("pressure", {}).items()
                                               if not k.startswith("_")}}
        self.vision_timeout_s = rules.get("vision_timeout_s", 1.0)
        self.inp = _Inputs(door=door, car=car)
        self.state = DOOR_OPEN if door == "open" else MONITORING
        self.since = start_ts
        self.warn_last_ts: Optional[float] = None
        self.warn_kinds: Set[str] = set()
        self.blocked = False
        self.start_ts = start_ts

    # ================================================================ 입력
    def on_pose(self, ts: float, level: int, reasons: List[str], leaning: bool = False,
                pet_near: bool = False, status: str = "ok") -> List[Any]:
        """PoseAnalyzer 결과 (안정 수준). status 는 입력 PoseFrame 의 status."""
        i = self.inp
        i.last_pose_ts = ts
        if status == "ok":
            i.vision_bad_since = None
            i.pose_level, i.pose_reasons, i.leaning, i.pet_near = level, list(reasons), leaning, pet_near
        else:
            i.vision_bad_since = i.vision_bad_since if i.vision_bad_since is not None else ts
            i.pose_level, i.pose_reasons, i.leaning, i.pet_near = 0, [], False, False
        return self._step(ts, f"pose:{level}")

    def on_pressure(self, msg: PressureSample) -> List[Any]:
        i = self.inp
        i.last_pressure_ts = msg.ts
        if i.p_baseline is None:
            i.p_baseline = msg.grams
        i.p_excess = msg.grams - i.p_baseline
        thr = self.p["threshold_g"]
        if thr is not None and i.p_excess >= thr:
            i.p_over_since = i.p_over_since if i.p_over_since is not None else msg.ts
        else:
            i.p_over_since = None
        return self._step(msg.ts, "pressure")

    def on_device(self, msg: DeviceEvent) -> List[Any]:
        i = self.inp
        out: List[Any] = []
        if msg.kind == "door":
            i.door = msg.value
            if msg.value == "open":
                i.p_baseline = None              # 문이 열리면 압력 영점 다시 잡기
                i.p_over_since = None
            if msg.value == "open" and self.state == DANGER_MOVING and self.r["danger_moving_reset"] == "door_open":
                out.append(ActionCommand(msg.ts, "car", "emergency_reset", {}))
        elif msg.kind == "car":
            i.car = msg.value
        elif msg.kind == "fault":
            if msg.value == "clear":
                i.device_faults.clear()
            else:
                i.device_faults.add(msg.value)
        elif msg.kind == "operator" and msg.value == "reset":
            if self.state == DANGER_MOVING:
                out.append(ActionCommand(msg.ts, "car", "emergency_reset", {}))   # ESP32 의 비상정지 래치 해제
                out += self._enter(msg.ts, self._target(msg.ts, ignore_latch=True), "operator_reset")
        return out + self._step(msg.ts, f"{msg.kind}:{msg.value}")

    def tick(self, now: float) -> List[Any]:
        """시간이 흘러서 생기는 변화 (비전 타임아웃, 경고 유지 끝남). 주기적으로 부른다."""
        return self._step(now, "tick")

    def handle(self, msg: Any) -> List[Any]:
        """메시지 하나 처리 (PressureSample / DeviceEvent). 자세는 on_pose 로 넣는다."""
        if isinstance(msg, PressureSample):
            return self.on_pressure(msg)
        if isinstance(msg, DeviceEvent):
            return self.on_device(msg)
        return []

    # ================================================================ 판단
    def faults(self, now: float) -> List[str]:
        i = self.inp
        out = sorted(f"device:{f}" for f in i.device_faults)
        t_ref = i.last_pose_ts if i.last_pose_ts is not None else self.start_ts
        if now - t_ref > self.vision_timeout_s:
            out.append("vision_timeout")
        elif i.vision_bad_since is not None and now - i.vision_bad_since > self.vision_timeout_s:
            out.append("vision_error")
        pt = self.p["timeout_s"]
        if pt is not None and self.p["threshold_g"] is not None:
            p_ref = i.last_pressure_ts if i.last_pressure_ts is not None else self.start_ts
            if now - p_ref > pt:
                out.append("pressure_timeout")
        return out

    def pressure_alert(self, now: float) -> bool:
        i = self.inp
        return i.p_over_since is not None and now - i.p_over_since >= (self.p["sustain_s"] or 0.0)

    def evidence(self, now: float) -> Dict[str, Any]:
        """지금 시점의 근거 요약 (StateChange.evidence 에도 들어감)."""
        i, r = self.inp, self.r
        p_alert = self.pressure_alert(now)
        danger = []
        if i.pose_level >= r["danger_pose_level"]:
            danger.append("pose_danger")
        if p_alert and i.pose_level >= r["warning_pose_level"]:
            danger.append("pose_caution+pressure")
        if p_alert and i.pet_near:
            danger.append("pet+pressure")
        warn = []
        if i.pose_level >= r["warning_pose_level"]:
            warn.append("pose")
        if p_alert:
            warn.append("pressure")
        if r["warn_on_leaning"] and i.leaning:
            warn.append("lean")
        if r["warn_on_pet_near_door"] and i.pet_near:
            warn.append("pet")
        return {"danger": danger, "warning": warn, "faults": self.faults(now), "door": i.door, "car": i.car,
                "pose_level": i.pose_level, "pose_reasons": i.pose_reasons,
                "pressure_excess_g": round(i.p_excess, 1)}

    def _target(self, now: float, ignore_latch: bool = False) -> str:
        i = self.inp
        ev = self.evidence(now)
        if not ignore_latch and self.state == DANGER_MOVING:
            if not (self.r["danger_moving_reset"] == "door_open" and i.door == "open"):
                return DANGER_MOVING                                   # 관리자 리셋까지 유지
        if i.door == "open":
            return FAULT if ev["faults"] else DOOR_OPEN
        if ev["danger"] or (not ignore_latch and self.state == DANGER_PRE):
            # DANGER_PRE 는 문이 열릴 때까지 유지. 그사이 카가 움직이면 DANGER_MOVING
            return DANGER_MOVING if i.car == "moving" else DANGER_PRE
        if ev["faults"]:
            return FAULT
        if ev["warning"]:
            self.warn_last_ts, self.warn_kinds = now, set(ev["warning"])
            return WARNING
        if self.warn_last_ts is not None and now - self.warn_last_ts <= self.r["warning_hold_s"]:
            return WARNING                                             # 경고 유지 시간
        return MONITORING

    # ================================================================ 전이·동작
    def _step(self, now: float, cause: str) -> List[Any]:
        out: List[Any] = []
        tgt = self._target(now)
        if tgt != self.state:
            out += self._enter(now, tgt, cause)
        out += self._update_block(now)
        return out

    def _enter(self, now: float, new: str, cause: str) -> List[Any]:
        prev, self.state, self.since = self.state, new, now
        ev = self.evidence(now)
        reason = {DANGER_MOVING: ",".join(ev["danger"]) or cause, DANGER_PRE: ",".join(ev["danger"]) or cause,
                  FAULT: ",".join(ev["faults"]) or cause, WARNING: ",".join(ev["warning"]) or "hold"}.get(new, cause)
        out: List[Any] = [StateChange(now, prev, new, reason, ev)]
        A = lambda target, action, **params: ActionCommand(now, target, action, params)
        if new in (DOOR_OPEN, MONITORING):
            self.warn_last_ts, self.warn_kinds = None, set()
            out += [A("buzzer", "off"), A("beacon", "off")]
        elif new == WARNING:
            kinds = set(ev["warning"]) or self.warn_kinds
            msg = VOICE["lean"] if kinds == {"lean"} else VOICE["pet"] if kinds == {"pet"} else VOICE["step_back"]
            out += [A("beacon", "yellow"), A("buzzer", "on", pattern="short"), A("voice", "say", text=msg)]
        elif new == DANGER_PRE:
            out += [A("door", "open"), A("buzzer", "on", pattern="continuous"), A("beacon", "red"),
                    A("voice", "say", text=VOICE["danger_pre"])]
        elif new == DANGER_MOVING:
            out += [A("car", "emergency_stop"), A("buzzer", "on", pattern="continuous"), A("beacon", "red"),
                    A("voice", "say", text=VOICE["danger_moving"])]
        elif new == FAULT:
            out += [A("beacon", "yellow", pattern="blink"), A("voice", "say", text=VOICE["fault"])]
        if new in (DANGER_PRE, DANGER_MOVING) and self.r["notify_on_danger"]:
            out.append(A("notify", "emergency", state=new, reason=reason))
        if new == FAULT:
            out.append(A("notify", "fault", reason=reason))
        return out

    def _update_block(self, now: float) -> List[Any]:
        """출발 보류: 서 있는 카를 출발시키지 않음. 상태 전이와 별개로 카 상태가 바뀔 때도 확인."""
        want = (self.state in (DANGER_PRE, FAULT) or
                (self.state == WARNING and bool(self.warn_kinds & set(self.r["block_on_warning"]))))
        if self.state == DANGER_MOVING:
            return []                            # 비상정지가 이미 카를 묶음 (해제는 리셋 때)
        if want and not self.blocked and self.inp.car == "stopped":
            self.blocked = True
            return [ActionCommand(now, "car", "block_start", {"reason": self.state})]
        if not want and self.blocked:
            self.blocked = False
            return [ActionCommand(now, "car", "block_end", {})]
        return []
