"""해석 파이프라인 — PoseAnalyzer(자세) + SafetyStateMachine(상태)을 한 곳에서 돌린다.

  PoseFrame ─────▶ PoseAnalyzer ──(안정 수준·기댐·반려동물)──▶ ┐
  PressureSample ───────────────────────────────────────────▶ ├─ SafetyStateMachine ─▶ StateChange, ActionCommand
  DeviceEvent ──(문 열림이면 PoseAnalyzer 초기화)────────────▶ ┘

handle(msg) 에 메시지를 순서대로 넣으면 내보낼 메시지 목록이 나온다. 실시간이면 tick(now) 도 주기적으로 부른다.
"""
from __future__ import annotations

from typing import Any, List, Optional

from door_common.door_geometry import DoorGeometry
from door_common.messages import DeviceEvent, PoseFrame, PressureSample

from .pose_features import PoseAnalyzer, PoseAssessment
from .state import SafetyStateMachine


class AnalysisPipeline:
    def __init__(self, door: DoorGeometry, rules: Optional[dict] = None, door_state: str = "open",
                 car_state: str = "stopped", start_ts: float = 0.0):
        self.rules = rules or {}
        self.pose = PoseAnalyzer(door, self.rules.get("pose"))
        self.sm = SafetyStateMachine(self.rules, door=door_state, car=car_state, start_ts=start_ts)
        self.last: Optional[PoseAssessment] = None      # 화면 표시용 최신 자세 해석

    @property
    def state(self) -> str:
        return self.sm.state

    def handle(self, msg: Any) -> List[Any]:
        if isinstance(msg, PoseFrame):
            a = self.pose.update(msg, self.rules.get("pet"))
            self.last = a
            leaning = any(p.flags.get("leaning_on_door") for p in a.persons)
            reasons = sorted({x for p in a.persons for x in p.reasons if not x.startswith("hold:")})
            return self.sm.on_pose(msg.ts, a.level, reasons, leaning, a.pet_near_door, msg.status)
        if isinstance(msg, DeviceEvent):
            if msg.kind == "door" and msg.value == "open":
                self.pose.reset()                         # 문이 열리면 끼일 수 없음 → 자세 판정 초기화
            return self.sm.on_device(msg)
        if isinstance(msg, PressureSample):
            return self.sm.on_pressure(msg)
        return []

    def tick(self, now: float) -> List[Any]:
        return self.sm.tick(now)
