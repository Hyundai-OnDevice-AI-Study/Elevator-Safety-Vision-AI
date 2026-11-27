"""가짜 ESP32 — Control/PROTOCOL.md 대로 움직이는 미니어처 엘리베이터 흉내.

펌웨어가 없어도 Pi 브리지·상태 머신 연동을 시험하기 위한 것이다. 동시에 '펌웨어가 이렇게 동작해야 한다'는
참고 구현이기도 하다 (loop 한 번에 조금씩 진행, delay 로 기다리지 않음).

입출력 없는 순수 클래스: feed_line(줄) · step(now) · press_go(층) 이 내보낼 줄 목록을 돌려준다.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .esp32_link import PROTO_VERSION, encode

DOOR_MOVE_S = 0.6        # 서보가 다 열리거나 닫히는 데 걸리는 시간 (센서 없는 안 A)
DOOR_DWELL_S = 2.0       # 다 열린 뒤 자동으로 닫기 시작할 때까지
FLOOR_TRAVEL_S = 1.0     # 한 층 이동 시간
HB_PERIOD_S = 0.2
PI_TIMEOUT_S = 0.6
MIN_FLOOR, MAX_FLOOR = 1, 5


class FakeEsp32:
    def __init__(self, now: float = 0.0, floor: int = 1):
        self.floor = floor
        self.target: Optional[int] = None         # 가려는 층 (문이 닫히면 출발)
        self.moving_dir = 0                       # +1 / -1 / 0
        self.next_floor_ts = 0.0
        self.door = "closed"                      # open / opening / closing / closed (opening 은 보고 안 함)
        self.door_ts = now                        # 문 동작 시작 또는 열림 완료 시각
        self.es = False
        self.blk = False
        self.last_pi_hb = now
        self.pi_alive = True
        self.last_seq: Optional[int] = None
        self.hb_seq = 0
        self.last_hb_tx: Optional[float] = None
        self.reported: Dict[str, str] = {}
        self.booted = False

    # ------------------------------------------------------------ 보고
    @property
    def car(self) -> str:
        return "moving" if self.moving_dir else "stopped"

    def _report(self, kind: str, value: str) -> List[str]:
        if self.reported.get(kind) == value:
            return []
        self.reported[kind] = value
        return [encode({"kind": kind, "value": value})]

    def _door_reported_value(self) -> Optional[str]:
        return {"open": "open", "closing": "closing", "closed": "closed"}.get(self.door)

    def _status_lines(self) -> List[str]:
        out = []
        dv = self._door_reported_value()
        if dv:
            out += self._report("door", dv)
        out += self._report("car", self.car)
        return out

    # ------------------------------------------------------------ Pi → ESP32
    def feed_line(self, line: str, now: float) -> List[str]:
        try:
            d: Dict[str, Any] = json.loads(line)
        except ValueError:
            return [f"# bad line {line.strip()!r}\n"]
        if "hb" in d:
            self.last_pi_hb = now
            if not self.pi_alive:
                self.pi_alive = True
                return ["# pi back\n"]
            return []
        seq = d.get("seq")
        if seq is not None and seq == self.last_seq:
            return [encode({"ack": seq})]           # 재전송: 다시 실행하지 않고 ack 만
        self.last_seq = seq
        t, a = d.get("target"), d.get("action")
        out: List[str] = []
        if (t, a) == ("car", "emergency_stop"):
            self.es = True
            self.moving_dir = 0
            self.target = None
            out += [encode({"ack": seq})] + self._report("car", "stopped") + ["# EMERGENCY STOP\n"]
        elif (t, a) == ("car", "emergency_reset"):
            self.es = False
            out.append(encode({"ack": seq}))
        elif (t, a) == ("car", "block_start"):
            self.blk = True
            out.append(encode({"ack": seq}))
        elif (t, a) == ("car", "block_end"):
            self.blk = False
            out.append(encode({"ack": seq}))
        elif (t, a) == ("door", "open"):
            if self.moving_dir:
                out.append(encode({"ack": seq, "ok": 0, "err": "car_moving"}))
            else:
                out.append(encode({"ack": seq}))
                if self.door == "open":
                    self.door_ts = now                  # 유지 시간 다시 셈
                elif self.door in ("closing", "closed"):
                    self.door, self.door_ts = "opening", now
        elif (t, a) == ("esp32", "report"):
            self.reported.clear()
            out += [encode({"ack": seq})] + self._status_lines()
        else:
            out.append(encode({"ack": seq, "ok": 0, "err": "unknown"}))
        return out

    # ------------------------------------------------------------ 버튼
    def press_go(self, floor: int, now: float) -> List[str]:
        if not MIN_FLOOR <= floor <= MAX_FLOOR:
            return [f"# bad floor {floor}\n"]
        if self.es or self.blk or not self.pi_alive or self.moving_dir:
            why = "es" if self.es else "blk" if self.blk else "no_pi" if not self.pi_alive else "moving"
            return [f"# go {floor} ignored ({why})\n"]
        if floor == self.floor:
            self.target = None
            if self.door in ("closed", "closing"):
                self.door, self.door_ts = "opening", now
            return [f"# already at {floor}\n"]
        self.target = floor
        if self.door in ("open", "opening"):
            self.door, self.door_ts = "closing", now
        return [f"# go {floor}\n"]

    # ------------------------------------------------------------ loop() 한 번
    def step(self, now: float) -> List[str]:
        out: List[str] = []
        if not self.booted:
            self.booted = True
            out += [encode({"hello": 1, "fw": "fake-0.1", "proto": PROTO_VERSION})] + self._status_lines()

        # Pi 감시
        if self.pi_alive and now - self.last_pi_hb > PI_TIMEOUT_S:
            self.pi_alive = False
            out.append("# pi lost: departures blocked\n")

        # 문
        if self.door == "opening" and now - self.door_ts >= DOOR_MOVE_S:
            self.door, self.door_ts = "open", now
        elif self.door == "open" and now - self.door_ts >= DOOR_DWELL_S:
            self.door, self.door_ts = "closing", now
        elif self.door == "closing" and now - self.door_ts >= DOOR_MOVE_S:
            self.door, self.door_ts = "closed", now

        # 카: 출발 (문이 닫혔고 막는 것이 없을 때. 출발 직전에 다시 확인)
        if self.target is not None and not self.moving_dir and self.door == "closed":
            if self.es or self.blk or not self.pi_alive:
                pass                                       # 막혀 있으면 대기 (풀리면 출발)
            else:
                self.moving_dir = 1 if self.target > self.floor else -1
                self.next_floor_ts = now + FLOOR_TRAVEL_S
        # 카: 이동 (Pi 가 끊겨도 이번 이동은 끝낸다)
        if self.moving_dir and now >= self.next_floor_ts:
            self.floor += self.moving_dir
            if self.floor == self.target:
                self.moving_dir, self.target = 0, None
                self.door, self.door_ts = "opening", now
            else:
                self.next_floor_ts = now + FLOOR_TRAVEL_S

        out += self._status_lines()

        if self.last_hb_tx is None or now - self.last_hb_tx >= HB_PERIOD_S:
            self.last_hb_tx = now
            self.hb_seq = self.hb_seq % 65535 + 1
            out.append(encode({"hb": self.hb_seq, "door": self._door_reported_value() or self.reported.get("door"),
                               "car": self.car, "blk": int(self.blk), "es": int(self.es), "fl": self.floor}))
        return out
