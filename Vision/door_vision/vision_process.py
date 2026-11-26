"""비전 프로세스 메인 루프.

  영상 파일 ─▶ YOLO-pose(ONNX) ─▶ 추적(track_id) ─▶ (N프레임마다) 반려동물 탐지 ─▶ PoseFrame 발행

설계 문서 원칙
  - 비전은 CPU를 가장 많이 쓰므로 '별도 프로세스'로 돈다. 판단 프로세스와는 메시지로만 연결.
  - 사람이 없어도, 영상이 끊겨도 PoseFrame 을 계속 보낸다(status 로 표시).
    → 판단 쪽은 'PoseFrame 이 일정 시간 안 오면 = 비전 멈춤 = 고장(출발 금지)'으로 판단할 수 있다.

실행 예
  # 모델 없이 Sim 으로 (메시지 흐름·판단 로직 개발용)
  python -m door_vision.vision_process --sim dragged --print

  # 실제 모델 + 영상 파일, 화면 표시, 기록
  python -m door_vision.vision_process --source videos/drag_01.mp4 --show --record out/drag_01.jsonl

  # Pi 두 대로 나눌 때: 판단 Pi 로 UDP 전송
  python -m door_vision.vision_process --source videos/drag_01.mp4 --udp 192.168.0.20:5600
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from typing import Optional

from door_common.messages import Person, PoseFrame, empty_pose_frame, now, round_kpts
from door_common.transport import JsonlRecorder, MultiPublisher, QueuePublisher, UdpPublisher
from door_common.door_geometry import DoorGeometry


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class FpsMeter:
    def __init__(self, n: int = 30):
        self.ts = deque(maxlen=n)

    def tick(self) -> float:
        self.ts.append(time.perf_counter())
        if len(self.ts) < 2:
            return 0.0
        return (len(self.ts) - 1) / (self.ts[-1] - self.ts[0])


def run_vision(cfg: dict, publisher, source: Optional[str] = None, sim: Optional[str] = None,
               show: bool = False, print_msgs: bool = False, max_frames: int = 0,
               analyze: bool = False, rules: Optional[dict] = None, stop_event=None) -> None:
    door = DoorGeometry.from_config(cfg)
    analyzer = None
    if analyze or show:
        # [통합 확인용] --show/--analyze 화면에 판단 수준을 색으로 보여 주려고 해석 코드를 잠깐 빌려 쓴다.
        # 비전 → 해석 의존은 여기 한 곳뿐. 해석 프로세스(state 등)가 생기면 메시지로 받도록 바꿀 예정.
        _ana = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Analysis")
        if _ana not in sys.path:
            sys.path.insert(0, _ana)
        from door_analysis.pose_features import PoseAnalyzer
        analyzer = PoseAnalyzer(door, (rules or {}).get("pose"))

    # ----------------------------------------------------------- Sim 모드
    if sim:
        from door_common.fake_pose import generate
        fps = cfg.get("sim_fps", 15)
        t0 = now()
        for pf in generate(sim, duration=cfg.get("sim_duration", 3.0), fps=fps, t0=t0,
                           with_pet=cfg.get("sim_with_pet", False)):
            while now() < pf.ts:                       # 실시간 속도로 재생
                time.sleep(0.002)
            publisher.publish(pf)
            if analyzer:
                a = analyzer.update(pf)
                if print_msgs:
                    print(f"{pf.frame_id:4d} t={pf.ts - t0:5.2f}s pose={a.level_name:8s} "
                          f"{[r for p in a.persons for r in p.reasons]}")
            elif print_msgs:
                print(pf.frame_id, len(pf.persons))
        return

    # ----------------------------------------------------------- 실제 모델
    import cv2

    from .pet_detector import PetDetector
    from .pose_estimator import PoseEstimator
    from .tracker import IouTracker
    from .video_source import VideoFileSource

    vc = cfg["vision"]
    pose = PoseEstimator(vc["pose_model"], vc.get("input_size"), vc.get("conf_thr", 0.4),
                         vc.get("iou_thr", 0.5), vc.get("max_persons", 5), vc.get("threads", 4))
    pet_cfg = cfg.get("pet", {})
    pets = None
    if pet_cfg.get("enabled") and os.path.exists(pet_cfg.get("model", "")):
        pets = PetDetector(pet_cfg["model"], pet_cfg.get("input_size"), pet_cfg.get("conf_thr", 0.35),
                           vc.get("threads", 4))
    tracker = IouTracker()
    src = VideoFileSource(source or vc["video"], realtime=vc.get("realtime", True),
                          loop=vc.get("loop", False), resize_width=vc.get("resize_width"))
    meter = FpsMeter()
    last_pets = []
    n = 0
    try:
        while not (stop_event and stop_event.is_set()):
            ok, frame, idx = src.read()
            if not ok:
                publisher.publish(empty_pose_frame(idx, "no_frame", src.path))
                break
            try:
                raw = pose(frame)
                ids = tracker.update([r.bbox for r in raw])
                persons = [Person(int(t), round(r.score, 3), [round(float(v), 4) for v in r.bbox],
                                  round_kpts(r.kpts.tolist())) for t, r in zip(ids, raw)]
                if pets and idx % pet_cfg.get("every_n", 3) == 0:
                    last_pets = pets(frame, door)
                pf = PoseFrame(ts=now(), frame_id=idx, persons=persons, pets=last_pets,
                               infer_ms=round(pose.last_timing["infer_ms"], 1), source=src.path,
                               img_w=int(frame.shape[1]), img_h=int(frame.shape[0]))
            except Exception as e:                                 # 추론 실패도 '고장'으로 알린다
                print("vision error:", e, file=sys.stderr)
                pf = empty_pose_frame(idx, "error", src.path)
            publisher.publish(pf)
            fps = meter.tick()
            a = analyzer.update(pf) if analyzer else None
            if print_msgs:
                lvl = a.level_name if a else "-"
                print(f"{idx:5d} persons={len(pf.persons)} pets={len(pf.pets)} "
                      f"infer={pf.infer_ms:5.1f}ms fps={fps:4.1f} pose={lvl} dropped={src.dropped}")
            if show:
                from .draw import draw_overlay
                cv2.imshow("vision", draw_overlay(frame, pf, door, assessment=a, fps=fps))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        src.release()
        if show:
            cv2.destroyAllWindows()


def start_vision_process(cfg: dict, queue: "mp.Queue", **kw) -> "mp.Process":
    """판단 프로세스(메인)에서 비전을 자식 프로세스로 띄울 때 사용.

    예)
        q = mp.Queue(maxsize=30)
        p = start_vision_process(cfg, q, source="videos/a.mp4")
        sub = QueueSubscriber(q)
        while True:
            for msg in sub.poll(timeout=0.1): ...
    """
    proc = mp.Process(target=run_vision, args=(cfg, QueuePublisher(queue)), kwargs=kw, daemon=True)
    proc.start()
    return proc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="엘리베이터 도어 끌림 감지 - 비전 프로세스")
    ap.add_argument("--config", default="config/vision.json")
    ap.add_argument("--rules", default="../Analysis/config/rules.json", help="--show/--analyze 때만 사용 (해석 팀 임계값)")
    ap.add_argument("--source", help="영상 파일 경로 (설정 파일 값 덮어쓰기)")
    ap.add_argument("--sim", help="가짜 자세 시나리오 이름 (모델 없이 실행)")
    ap.add_argument("--show", action="store_true", help="디버그 화면 표시")
    ap.add_argument("--print", dest="print_msgs", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="자세 해석 결과도 함께 출력")
    ap.add_argument("--record", help="PoseFrame 을 jsonl 로 저장")
    ap.add_argument("--udp", help="host:port 로 PoseFrame 전송")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = load_json(a.config)
    rules = load_json(a.rules) if os.path.exists(a.rules) else {}
    if not a.sim:                                   # 실제 모드: 기록 파일을 만들기 전에 준비물 확인
        model = cfg["vision"]["pose_model"]
        video = a.source or cfg["vision"]["video"]
        missing = [f"  - 포즈 모델: {model}  → PC에서 tools/export_onnx.py 로 만든 뒤 복사"
                   for _ in [0] if not os.path.exists(model)]
        missing += [f"  - 영상: {video}" for _ in [0] if not os.path.exists(video)]
        if missing:
            print("실행할 수 없습니다. 아래 파일이 없습니다:\n" + "\n".join(missing) +
                  "\n모델 없이 흐름만 보려면: python -m door_vision.vision_process --sim leash_yank --print",
                  file=sys.stderr)
            sys.exit(1)
    if a.sim:
        from door_common.fake_pose import SIM_DOOR_CONFIG
        cfg = {**cfg, **SIM_DOOR_CONFIG}
    rec = JsonlRecorder(a.record) if a.record else None
    udp = None
    if a.udp:
        host, port = a.udp.split(":")
        udp = UdpPublisher(host, int(port))

    class _Null:
        def publish(self, m):
            pass

    pub = MultiPublisher(rec, udp) if (rec or udp) else _Null()
    try:
        run_vision(cfg, pub, a.source, a.sim, a.show, a.print_msgs, a.max_frames,
                   analyze=a.analyze or bool(a.sim), rules=rules)
    finally:
        if rec:
            rec.close()


if __name__ == "__main__":
    main()
