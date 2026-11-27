"""ESP32 시리얼 프로토콜 (Control/PROTOCOL.md) 의 Pi 쪽 처리 — 입출력 없는 순수 클래스.

줄(str)을 넣으면 DeviceEvent 가 나오고, ActionCommand 를 넣으면 보낼 줄(str)이 나온다.
시리얼 읽기/쓰기는 esp32_bridge.py 가 한다. 그래서 테스트·가짜 ESP32 에서 같은 코드를 쓴다.
시간은 호출하는 쪽이 넘기는 now (time.monotonic 초) 만 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from door_common.messages import ActionCommand, DeviceEvent

PROTO_VERSION = 1

ESP32_TARGETS = ("door", "car", "esp32")          # 이 target 의 ActionCommand 만 ESP32 로 보낸다
ALLOWED = {
    "door": {"open"},
    "car": {"emergency_stop", "emergency_reset", "block_start", "block_end"},
    "esp32": {"report"},
}
DOOR_VALUES = {"open", "closing", "closed"}
CAR_VALUES = {"stopped", "moving"}

DEFAULTS = {
    "hb_period_s": 0.2,          # Pi → ESP32 하트비트 간격
    "link_timeout_s": 0.6,       # ESP32 하트비트가 이만큼 없으면 연결 끊김
    "ack_timeout_s": 0.1,        # ack 가 이만큼 안 오면 재전송
    "max_retries": 3,            # 재전송 횟수. 넘으면 fault esp32_no_ack
}

LINK_FAULT = "esp32_link"
NO_ACK_FAULT = "esp32_no_ack"


def encode(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":")) + "\n"


@dataclass
class _Pending:
    seq: int
    line: str
    sent_ts: float
    tries: int


class Esp32Link:
    def __init__(self, start_ts: float = 0.0, **cfg):
        self.cfg = {**DEFAULTS, **cfg}
        self.seq = 0
        self.hb_seq = 0
        self.pending: Dict[int, _Pending] = {}
        self.last_rx_hb: float = start_ts          # 처음에는 시작 시각 기준으로 기다림
        self.last_tx_hb: Optional[float] = None
        self.link_up = True                        # 처음엔 끊김 보고 안 함 → 타임아웃 지나면 down
        self.door: Optional[str] = None            # 마지막으로 Pi 가 아는 ESP32 상태
        self.car: Optional[str] = None
        self.snapshot: Dict[str, Any] = {}         # 마지막 하트비트 내용 (blk, es, fl …)
        self.hello: Optional[Dict[str, Any]] = None
        self.log: List[Tuple[float, str]] = []     # 거부·버린 줄 등 (사람이 볼 것)
        self.hb_enabled = True                     # 끊김 시험용 (브리지의 quiet)

    # ------------------------------------------------------------ ESP32 → Pi
    def feed_line(self, line: str, now: float) -> List[DeviceEvent]:
        """받은 줄 하나 처리. 상태 머신에 넣을 DeviceEvent 목록을 돌려준다."""
        line = line.strip()
        if not line:
            return []
        if line.startswith("#"):
            self.log.append((now, line))
            return []
        try:
            d = json.loads(line)
            if not isinstance(d, dict):
                raise ValueError
        except ValueError:
            self.log.append((now, f"bad line: {line!r}"))
            return []

        out: List[DeviceEvent] = []
        if "hb" in d:
            self.last_rx_hb = now
            self.snapshot = d
            out += self._sync("door", d.get("door"), now)
            out += self._sync("car", d.get("car"), now)
        elif "hello" in d:
            self.last_rx_hb = now
            self.hello = d
            self.door = self.car = None              # ESP32 재부팅 → 이어서 오는 상태를 그대로 받음
            if d.get("proto") not in (None, PROTO_VERSION):
                self.log.append((now, f"proto mismatch: esp32={d.get('proto')} pi={PROTO_VERSION}"))
        elif "ack" in d:
            p = self.pending.pop(d["ack"], None)
            if d.get("ok", 1) in (0, False):
                self.log.append((now, f"nack seq={d['ack']} err={d.get('err')} line={p.line.strip() if p else '?'}"))
        elif "kind" in d and "value" in d:
            if d["kind"] in ("door", "car"):
                out += self._sync(d["kind"], d["value"], now)
            elif d["kind"] == "fault":
                out.append(DeviceEvent(now, "fault", str(d["value"])))
            else:
                self.log.append((now, f"unknown kind: {line}"))
        else:
            self.log.append((now, f"unknown line: {line}"))

        if not self.link_up and now - self.last_rx_hb <= self.cfg["link_timeout_s"]:
            self.link_up = True
            out.append(DeviceEvent(now, "fault", "clear"))
            self.log.append((now, "link up"))
        return out

    def _sync(self, kind: str, value: Optional[str], now: float) -> List[DeviceEvent]:
        valid = DOOR_VALUES if kind == "door" else CAR_VALUES
        if value is None:
            return []
        if value not in valid:
            self.log.append((now, f"bad {kind} value: {value}"))
            return []
        if getattr(self, kind) == value:
            return []
        setattr(self, kind, value)
        return [DeviceEvent(now, kind, value)]

    # ------------------------------------------------------------ Pi → ESP32
    def send(self, cmd: ActionCommand, now: float) -> Optional[str]:
        """ESP32 로 보낼 명령이면 줄을 만들어 돌려준다 (재전송 대기열에도 넣음). 아니면 None."""
        if cmd.target not in ESP32_TARGETS:
            return None
        if cmd.action not in ALLOWED[cmd.target]:
            raise ValueError(f"프로토콜에 없는 명령: {cmd.target}/{cmd.action}")
        self.seq = self.seq % 65535 + 1
        line = encode({"seq": self.seq, "target": cmd.target, "action": cmd.action})
        self.pending[self.seq] = _Pending(self.seq, line, now, 1)
        return line

    def tick(self, now: float) -> Tuple[List[str], List[DeviceEvent]]:
        """시간이 흘러 생기는 일: 하트비트 송신, 재전송, 연결 끊김. (보낼 줄, 상태 머신에 넣을 이벤트)"""
        lines: List[str] = []
        events: List[DeviceEvent] = []
        c = self.cfg
        if self.hb_enabled and (self.last_tx_hb is None or now - self.last_tx_hb >= c["hb_period_s"]):
            self.hb_seq = self.hb_seq % 65535 + 1
            self.last_tx_hb = now
            lines.append(encode({"hb": self.hb_seq}))
        for seq, p in list(self.pending.items()):
            if now - p.sent_ts < c["ack_timeout_s"]:
                continue
            if p.tries > c["max_retries"]:
                del self.pending[seq]
                self.log.append((now, f"no ack seq={seq} line={p.line.strip()}"))
                events.append(DeviceEvent(now, "fault", NO_ACK_FAULT))
            else:
                p.tries += 1
                p.sent_ts = now
                lines.append(p.line)
        if self.link_up and now - self.last_rx_hb > c["link_timeout_s"]:
            self.link_up = False
            self.log.append((now, "link down"))
            events.append(DeviceEvent(now, "fault", LINK_FAULT))
        return lines, events
