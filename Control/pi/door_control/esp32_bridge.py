"""ESP32 시리얼 브리지 — 시리얼 입출력 + Esp32Link.

    bridge = Esp32Bridge(port="/dev/ttyUSB0")     # 또는 Esp32Bridge(fake=True)
    events = bridge.poll()                        # DeviceEvent 목록 → 상태 머신 handle()
    bridge.send(action_command)                   # target 이 door/car 인 ActionCommand

poll() 을 20~50 ms 마다 부른다 (하트비트 송신·재전송·연결 감시가 여기서 돈다).

손으로 시험하기:
    python -m door_control.esp32_bridge --fake
    python -m door_control.esp32_bridge --port /dev/ttyUSB0 [-v]
명령: estop | reset | block | unblock | open | report | go <층>(가짜만) | quiet | help | q
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from typing import List, Optional

from door_common.messages import ActionCommand, DeviceEvent, now

from .esp32_link import Esp32Link
from .fake_esp32 import FakeEsp32


class FakeTransport:
    """가짜 ESP32 를 스레드로 돌리며 시리얼처럼 readline/write 를 제공 (하드웨어 없이 시험)."""

    def __init__(self, period_s: float = 0.01):
        self.fake = FakeEsp32(now())
        self._rx: "queue.Queue[bytes]" = queue.Queue()
        self._lock = threading.Lock()
        self._stop = False
        self._period = period_s
        threading.Thread(target=self._run, daemon=True).start()

    def _push(self, lines: List[str]) -> None:
        for ln in lines:
            self._rx.put(ln.encode())

    def _run(self) -> None:
        while not self._stop:
            with self._lock:
                self._push(self.fake.step(now()))
            time.sleep(self._period)

    def write(self, data: bytes) -> None:
        with self._lock:
            for ln in data.decode().splitlines():
                self._push(self.fake.feed_line(ln, now()))

    def press_go(self, floor: int) -> None:
        with self._lock:
            self._push(self.fake.press_go(floor, now()))

    def readline(self, timeout: float = 0.1) -> bytes:
        try:
            return self._rx.get(timeout=timeout)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self._stop = True


class Esp32Bridge:
    def __init__(self, port: Optional[str] = None, baud: int = 115200, fake: bool = False, verbose: bool = False,
                 **link_cfg):
        if fake:
            self.io = FakeTransport()
            self._readline = self.io.readline
        else:
            import serial  # pyserial
            self.io = serial.Serial(port, baud, timeout=0.1)
            self._readline = self.io.readline
        self.link = Esp32Link(start_ts=now(), **link_cfg)
        self.verbose = verbose
        self._rx: "queue.Queue[tuple]" = queue.Queue()
        self._wlock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = b""
        while not self._closed:
            try:
                chunk = self._readline()
            except Exception as e:                      # 케이블 뽑힘 등 → 연결 감시가 fault 로 처리
                self._rx.put((now(), f"# serial error: {e}"))
                time.sleep(0.5)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._rx.put((now(), line.decode("ascii", "replace")))   # ★ 받은 순간 Pi 시각

    def _write(self, line: str) -> None:
        with self._wlock:
            self.io.write(line.encode())
        if self.verbose:
            print(f"  tx {line.strip()}")

    def poll(self) -> List[DeviceEvent]:
        """받은 줄 처리 + 하트비트·재전송·연결 감시. 상태 머신에 넣을 DeviceEvent 를 돌려준다."""
        events: List[DeviceEvent] = []
        while True:
            try:
                ts, line = self._rx.get_nowait()
            except queue.Empty:
                break
            if self.verbose:
                print(f"  rx {line}")
            events += self.link.feed_line(line, ts)
        lines, evs = self.link.tick(now())
        for ln in lines:
            self._write(ln)
        return events + evs

    def send(self, cmd: ActionCommand) -> bool:
        """ESP32 로 가는 명령이면 보내고 True. (buzzer 등 다른 target 이면 False)"""
        line = self.link.send(cmd, now())
        if line is None:
            return False
        self._write(line)
        return True

    def close(self) -> None:
        self._closed = True
        self.io.close()


# ---------------------------------------------------------------- 손으로 시험
SHORT = {
    "estop": ("car", "emergency_stop"), "reset": ("car", "emergency_reset"),
    "block": ("car", "block_start"), "unblock": ("car", "block_end"),
    "open": ("door", "open"), "report": ("esp32", "report"),
}

HELP = "명령: estop | reset | block | unblock | open | report | go <층>(가짜만) | quiet(하트비트 멈춤/재개) | q"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ESP32 시리얼 브리지 시험")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--port", help="예: /dev/ttyUSB0, COM5")
    g.add_argument("--fake", action="store_true", help="가짜 ESP32 와 대화")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("-v", "--verbose", action="store_true", help="주고받는 줄을 모두 찍음 (하트비트 포함)")
    a = ap.parse_args(argv)

    br = Esp32Bridge(port=a.port, baud=a.baud, fake=a.fake, verbose=a.verbose)
    t0 = now()
    cmds: "queue.Queue[str]" = queue.Queue()

    def stdin_reader():
        for ln in sys.stdin:
            cmds.put(ln.strip())
        cmds.put("q")
    threading.Thread(target=stdin_reader, daemon=True).start()
    print(HELP)

    n_log, last_snap = 0, None
    try:
        while True:
            for ev in br.poll():
                print(f"[{ev.ts - t0:7.2f}] DeviceEvent {ev.kind}={ev.value}")
            for ts, msg in br.link.log[n_log:]:
                print(f"[{ts - t0:7.2f}] {msg}")
            n_log = len(br.link.log)
            snap = {k: v for k, v in br.link.snapshot.items() if k != "hb"}
            if snap and snap != last_snap:
                print(f"           hb {snap}")
                last_snap = snap
            try:
                c = cmds.get_nowait()
            except queue.Empty:
                time.sleep(0.02)
                continue
            parts = c.split()
            if not parts:
                continue
            if parts[0] in ("q", "quit", "exit"):
                break
            if parts[0] in SHORT:
                tgt, act = SHORT[parts[0]]
                br.send(ActionCommand(now(), tgt, act, {}))
                print(f"[{now() - t0:7.2f}] send {tgt}/{act} seq={br.link.seq}")
            elif parts[0] == "go" and len(parts) == 2 and a.fake:
                br.io.press_go(int(parts[1]))
            elif parts[0] == "quiet":
                br.link.hb_enabled = not br.link.hb_enabled
                print(f"           Pi 하트비트 {'재개' if br.link.hb_enabled else '멈춤'}")
            else:
                print(HELP)
    except KeyboardInterrupt:
        pass
    finally:
        br.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
