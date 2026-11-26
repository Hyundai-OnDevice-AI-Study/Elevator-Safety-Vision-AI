"""모듈 간 메시지 5종 정의 (설계 문서 2장).

모듈끼리는 이 5가지만 주고받는다:
  1) PoseFrame       : 비전 → 판단      (프레임별 사람 자세 + 반려동물)
  2) PressureSample  : 센싱 → 판단      (로드셀 압력값)
  3) DeviceEvent     : 센싱 → 판단      (문·카·고장 이벤트)
  4) StateChange     : 판단 → 제어/모니터링
  5) ActionCommand   : 판단 → 제어

좌표는 모두 0~1 정규화 좌표(이미지 폭/높이로 나눈 값)로 보낸다.
해상도가 바뀌거나 영상이 바뀌어도 판단 코드의 임계값이 그대로 유지되도록 하기 위함이다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def now() -> float:
    """모든 모듈이 공유하는 시간 기준 (단조 증가 시계, 초)."""
    return time.monotonic()


# ---------------------------------------------------------------- 1. PoseFrame
@dataclass
class Person:
    track_id: int
    score: float
    bbox: List[float]                 # [x1, y1, x2, y2] 정규화
    keypoints: List[List[float]]      # 17 x [x, y, conf] (COCO 순서) 정규화


@dataclass
class PetDetection:
    label: str                        # "cat" | "dog"
    score: float
    bbox: List[float]                 # [x1, y1, x2, y2] 정규화
    in_door_zone: bool = False        # 문 주변 영역 안에 있는가


@dataclass
class PoseFrame:
    ts: float
    frame_id: int
    persons: List[Person] = field(default_factory=list)
    pets: List[PetDetection] = field(default_factory=list)
    status: str = "ok"                # ok | no_frame | error  (판단 쪽 '고장' 판정용)
    infer_ms: float = 0.0
    source: str = ""
    img_w: int = 0                    # 원본 영상 크기. 정규화 좌표는 가로·세로 배율이 달라서
    img_h: int = 0                    # 거리 계산 전에 화면비(w/h)로 되돌려야 한다
    type: str = "PoseFrame"
    v: int = SCHEMA_VERSION

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PoseFrame":
        return PoseFrame(
            ts=d["ts"], frame_id=d["frame_id"],
            persons=[Person(**p) for p in d.get("persons", [])],
            pets=[PetDetection(**p) for p in d.get("pets", [])],
            status=d.get("status", "ok"), infer_ms=d.get("infer_ms", 0.0),
            source=d.get("source", ""), img_w=d.get("img_w", 0), img_h=d.get("img_h", 0),
        )


# ----------------------------------------------------------- 2. PressureSample
@dataclass
class PressureSample:
    ts: float
    raw: int                          # HX711 원시값
    grams: float                      # 영점·스케일 보정 후 값
    type: str = "PressureSample"
    v: int = SCHEMA_VERSION

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PressureSample":
        return PressureSample(ts=d["ts"], raw=d["raw"], grams=d["grams"])


# -------------------------------------------------------------- 3. DeviceEvent
@dataclass
class DeviceEvent:
    ts: float
    kind: str                         # "door" | "car" | "fault"
    value: str                        # door: open/closing/closed, car: stopped/moving, fault: 원인
    type: str = "DeviceEvent"
    v: int = SCHEMA_VERSION

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DeviceEvent":
        return DeviceEvent(ts=d["ts"], kind=d["kind"], value=d["value"])


# -------------------------------------------------------------- 4. StateChange
@dataclass
class StateChange:
    ts: float
    prev: str
    state: str                        # DOOR_OPEN | MONITORING | WARNING | DANGER_PRE | DANGER_MOVING | FAULT
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    type: str = "StateChange"
    v: int = SCHEMA_VERSION

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StateChange":
        return StateChange(ts=d["ts"], prev=d["prev"], state=d["state"],
                           reason=d.get("reason", ""), evidence=d.get("evidence", {}))


# ------------------------------------------------------------ 5. ActionCommand
@dataclass
class ActionCommand:
    ts: float
    target: str                       # door | car | buzzer | beacon | voice
    action: str                       # open / block_start / emergency_stop / on / off / yellow / red ...
    params: Dict[str, Any] = field(default_factory=dict)
    type: str = "ActionCommand"
    v: int = SCHEMA_VERSION

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ActionCommand":
        return ActionCommand(ts=d["ts"], target=d["target"], action=d["action"],
                             params=d.get("params", {}))


# ------------------------------------------------------------ 직렬화 도우미
_TYPES = {
    "PoseFrame": PoseFrame,
    "PressureSample": PressureSample,
    "DeviceEvent": DeviceEvent,
    "StateChange": StateChange,
    "ActionCommand": ActionCommand,
}


def to_json(msg: Any) -> str:
    return json.dumps(asdict(msg), ensure_ascii=False, separators=(",", ":"))


def from_json(text: str) -> Any:
    d = json.loads(text)
    cls = _TYPES.get(d.get("type"))
    if cls is None:
        raise ValueError(f"알 수 없는 메시지 타입: {d.get('type')}")
    return cls.from_dict(d)


def round_kpts(kpts: List[List[float]], nd: int = 4) -> List[List[float]]:
    """전송량을 줄이기 위해 소수점 자리수를 제한."""
    return [[round(x, nd), round(y, nd), round(c, 3)] for x, y, c in kpts]


def empty_pose_frame(frame_id: int, status: str, source: str = "") -> PoseFrame:
    return PoseFrame(ts=now(), frame_id=frame_id, status=status, source=source)


__all__ = [
    "Person", "PetDetection", "PoseFrame", "PressureSample", "DeviceEvent",
    "StateChange", "ActionCommand", "to_json", "from_json", "now", "round_kpts",
    "empty_pose_frame",
]
