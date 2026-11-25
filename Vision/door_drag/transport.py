"""메시지 전송 계층.

- QueueBus : 같은 Pi 안에서 프로세스 간 전송 (multiprocessing.Queue)
- UdpPublisher / UdpSubscriber : 나중에 Pi를 두 대로 나눌 때 (비전 Pi → 판단 Pi)

판단 코드는 subscriber.poll() 만 부르므로, 전송 방식을 바꿔도 판단 코드는 그대로다.
"""
from __future__ import annotations

import queue
import socket
import time
from typing import Any, List, Optional

from .messages import from_json, to_json


class QueuePublisher:
    def __init__(self, q):
        self.q = q
        self.dropped = 0

    def publish(self, msg: Any) -> None:
        text = to_json(msg)
        # 소비자가 느리면 가장 오래된 것을 버리고 최신을 넣는다 (지연보다 최신성이 중요)
        for _ in range(5):
            try:
                self.q.put_nowait(text)
                return
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    time.sleep(0.001)
        self.dropped += 1


class QueueSubscriber:
    def __init__(self, q):
        self.q = q

    def poll(self, timeout: float = 0.0, max_items: int = 100) -> List[Any]:
        out = []
        try:
            out.append(from_json(self.q.get(timeout=timeout) if timeout else self.q.get_nowait()))
            while len(out) < max_items:
                out.append(from_json(self.q.get_nowait()))
        except queue.Empty:
            pass
        return out


class UdpPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 5600):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, msg: Any) -> None:
        data = to_json(msg).encode("utf-8")
        if len(data) > 60000:
            raise ValueError("UDP 한 패킷에 담기에는 메시지가 너무 큽니다")
        self.sock.sendto(data, self.addr)


class UdpSubscriber:
    def __init__(self, host: str = "0.0.0.0", port: int = 5600):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)

    def poll(self, timeout: float = 0.0, max_items: int = 100) -> List[Any]:
        out = []
        self.sock.settimeout(timeout if timeout else None)
        if not timeout:
            self.sock.setblocking(False)
        try:
            while len(out) < max_items:
                data, _ = self.sock.recvfrom(65535)
                out.append(from_json(data.decode("utf-8")))
                self.sock.setblocking(False)
        except (BlockingIOError, socket.timeout):
            pass
        return out


class JsonlRecorder:
    """PoseFrame 등을 한 줄에 하나씩 저장 → 나중에 evaluate.py로 재생/채점."""

    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def publish(self, msg: Any) -> None:
        self.f.write(to_json(msg) + "\n")

    def close(self) -> None:
        self.f.close()


class MultiPublisher:
    def __init__(self, *pubs):
        self.pubs = [p for p in pubs if p is not None]

    def publish(self, msg: Any) -> None:
        for p in self.pubs:
            p.publish(msg)


def read_jsonl(path: str) -> List[Any]:
    with open(path, encoding="utf-8") as f:
        return [from_json(line) for line in f if line.strip()]
