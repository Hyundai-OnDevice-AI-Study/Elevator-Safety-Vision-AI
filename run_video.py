"""노트북에서 영상 하나를 골라 비전 → 해석 → 상태 머신 결과를 화면으로 본다 (통합 확인용).

  python run_video.py Vision/videos/leash/leash_entrap_001.mp4
  python run_video.py Vision/videos/leash/cam03_leash_drag.mp4 --events cam03
  python run_video.py <영상> --config Vision/config/vision_cam03.json --door open --car stopped
  python run_video.py <영상> --events '[[0.6,"door","closing"],[1.25,"door","closed"],[1.5,"car","moving"]]'
  python run_video.py <영상> --save out/result.mp4          # 화면 없이 결과 영상만 저장
  python run_video.py <영상> --realtime                     # Pi 처럼 처리가 느리면 프레임을 버림

화면
  - 사람 뼈대: 자세 수준 색 (초록 NORMAL / 노랑 OBSERVE / 주황 CAUTION / 빨강 DANGER), 손목에 L/R 표시
  - 자홍 선: 문틈, 하늘색: 버튼 패널, 주황색: 무시 영역(거울)
  - 아래 띠: 상태 머신 상태, 문·카 상태, 판단 근거, 최근에 나간 제어 명령

키
  space 일시정지/재생   d 한 프레임 앞으로(일시정지 중)   q / ESC 종료   s 현재 화면 png 저장
  o 문 열림   c 문 닫힘   m 카 이동/정지 바꾸기   r 관리자 리셋        (문·카 이벤트를 손으로 넣기)

문·카 상태는 영상만으로 알 수 없으므로 --door/--car 로 시작 상태를 주거나, --events 로 시각별 이벤트를 주거나,
재생 중에 키로 넣는다. --events 에 labels.json 의 name(예: cam03)을 주면 그 항목의 device_events 를 쓴다.
문틈 좌표는 --config 의 "door". 카메라 구도가 다른 영상은 Vision/tools/draw_door_line.py 로 설정 파일을 먼저 만든다.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ("common", "Vision", "Analysis"):
    if os.path.join(ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, _d))

import cv2  # noqa: E402

from door_analysis.pipeline import AnalysisPipeline  # noqa: E402
from door_common import keypoints as K  # noqa: E402
from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_common.messages import ActionCommand, DeviceEvent, Person, PoseFrame, StateChange, round_kpts  # noqa: E402
from door_vision.draw import draw_overlay  # noqa: E402
from door_vision.pose_estimator import PoseEstimator  # noqa: E402
from door_vision.tracker import IouTracker  # noqa: E402

STATE_COLORS = {"DOOR_OPEN": (170, 170, 170), "MONITORING": (80, 200, 80), "WARNING": (0, 200, 255),
                "DANGER_PRE": (0, 0, 255), "DANGER_MOVING": (0, 0, 255), "FAULT": (255, 128, 0)}
LABELS = os.path.join(ROOT, "Analysis", "config", "scenarios", "labels.json")


def vision_path(p: str) -> str:
    """비전 설정 안의 상대 경로는 Vision/ 기준."""
    return p if os.path.isabs(p) else os.path.join(ROOT, "Vision", p)


def load_events(arg: str):
    if not arg:
        return []
    if arg.strip().startswith("["):
        return json.loads(arg)
    for item in json.load(open(LABELS, encoding="utf-8")):
        if item["name"] == arg or item["name"].startswith(arg):
            return item.get("device_events", [])
    raise SystemExit(f"--events: JSON 도 아니고 labels.json 에 '{arg}' 도 없습니다")


def text(img, s, org, scale, col, th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="영상으로 비전·해석·상태 머신 결과 보기")
    ap.add_argument("video", help="테스트할 영상 경로 (예: Vision/videos/leash/leash_entrap_001.mp4)")
    ap.add_argument("--config", default=os.path.join(ROOT, "Vision", "config", "vision_cam03.json"),
                    help="비전 설정 (모델·문틈 좌표). 기본: cam03 카메라")
    ap.add_argument("--rules", default=os.path.join(ROOT, "Analysis", "config", "rules.json"))
    ap.add_argument("--door", default="closed", choices=["open", "closing", "closed"], help="시작할 때 문 상태")
    ap.add_argument("--car", default="stopped", choices=["stopped", "moving"], help="시작할 때 카 상태")
    ap.add_argument("--events", help='시각별 문·카 이벤트 JSON [[초, kind, value], ...] 또는 labels.json 의 name')
    ap.add_argument("--realtime", action="store_true", help="처리가 느리면 프레임을 버림 (Pi 재현). 기본은 모든 프레임 처리")
    ap.add_argument("--width", type=int, default=1280, help="화면 폭")
    ap.add_argument("--save", help="화면 대신 결과 영상(mp4) 저장")
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    rules = json.load(open(a.rules, encoding="utf-8")) if os.path.exists(a.rules) else {}
    vc = cfg["vision"]
    door = DoorGeometry.from_config(cfg)
    pose = PoseEstimator(vision_path(vc["pose_model"]), vc.get("input_size"), vc.get("conf_thr", 0.4),
                         vc.get("iou_thr", 0.5), vc.get("max_persons", 5), vc.get("threads", 4))
    tracker = IouTracker()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    events = sorted(load_events(a.events), key=lambda e: e[0])
    door0 = next((v for _, k, v in events if k == "door"), a.door) if events and events[0][0] <= 0 else a.door
    pl = AnalysisPipeline(door, rules, door_state=door0, car_state=a.car, start_ts=0.0)
    dev = {"door": door0, "car": a.car}
    actions = deque(maxlen=5)
    log_lines = []

    def feed(msgs, t):
        for m in msgs:
            if isinstance(m, StateChange):
                line = f"{t:6.2f}s  {m.prev} -> {m.state}  ({m.reason})"
                print(line)
                log_lines.append(line)
            elif isinstance(m, ActionCommand):
                extra = " ".join(f"{k}={v}" for k, v in m.params.items() if k != "text")
                actions.append(f"{t:5.2f}s {m.target} {m.action} {extra}".rstrip())
                print(f"{'':10s}└ {m.target} {m.action} {extra}")

    def device(kind, value, t):
        dev[kind] = value
        print(f"{t:6.2f}s  [이벤트] {kind} = {value}")
        feed(pl.handle(DeviceEvent(t, kind, value)), t)

    writer = None
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)) or ".", exist_ok=True)
    else:
        cv2.namedWindow("run_video", cv2.WINDOW_AUTOSIZE)
    print(f"영상 {a.video}  {n_frames}프레임 {fps:.0f}fps  설정 {os.path.relpath(a.config, ROOT)}")
    print("키: space 정지/재생, d 한 프레임, o/c 문 열림/닫힘, m 카 이동/정지, r 리셋, s png, q 종료")

    idx, k_ev, paused, step_once, wall0 = -1, 0, False, False, time.monotonic()
    frame = None
    while True:
        if not paused or step_once or frame is None:
            step_once = False
            if a.realtime and idx >= 0:                       # 늦어진 만큼 프레임 버리기
                target = int((time.monotonic() - wall0) * fps)
                while idx < target - 1 and cap.grab():
                    idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            rw = vc.get("resize_width")                       # 비전 프로세스와 같은 입력 크기
            if rw and frame.shape[1] != rw:
                frame = cv2.resize(frame, (rw, int(frame.shape[0] * rw / frame.shape[1])))
            t = idx / fps                                     # 영상 시간 기준 (처리 속도와 무관)
            while k_ev < len(events) and events[k_ev][0] <= t:
                device(events[k_ev][1], events[k_ev][2], events[k_ev][0])
                k_ev += 1
            t_inf = time.perf_counter()
            raw = pose(frame)
            ids = tracker.update([r.bbox for r in raw])
            persons = [Person(int(i), round(r.score, 3), [round(float(v), 4) for v in r.bbox],
                              round_kpts(r.kpts.tolist())) for i, r in zip(ids, raw)]
            pf = PoseFrame(ts=t, frame_id=idx, persons=persons, infer_ms=(time.perf_counter() - t_inf) * 1000,
                           source=a.video, img_w=int(frame.shape[1]), img_h=int(frame.shape[0]))
            feed(pl.handle(pf), t)

            # ---- 그리기
            vis = draw_overlay(frame, pf, door, assessment=pl.last)
            h, w = vis.shape[:2]
            for p in pf.persons:
                for name, i in (("L", K.L_WRIST), ("R", K.R_WRIST)):
                    x, y, c = p.keypoints[i]
                    if c >= 0.35:
                        text(vis, name, (int(x * w) + 6, int(y * h)), 0.6, (0, 255, 255), 2)
            scale = a.width / w
            vis = cv2.resize(vis, (a.width, int(h * scale)))
            bar = 30 + 22 * actions.maxlen                    # 영상 아래에 붙이는 정보 띠 (영상을 가리지 않게)
            vis = cv2.copyMakeBorder(vis, 0, bar + 40, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            H = vis.shape[0]
            st = pl.state
            text(vis, f"{st}", (10, H - bar - 12), 0.9, STATE_COLORS[st], 2)
            reasons = sorted({x for p in pl.last.persons for x in p.reasons if not x.startswith("hold")}) if pl.last else []
            text(vis, f"t={t:5.2f}s  door={dev['door']}  car={dev['car']}  pose={pl.last.level_name if pl.last else '-'}  "
                      f"{','.join(reasons)[:80]}", (260, H - bar - 14), 0.5, (255, 255, 255))
            for j, s in enumerate(actions):
                text(vis, s, (10, H - bar + 18 + 22 * j), 0.5, (200, 200, 200))
            if writer is None and a.save:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], H))
            if writer is not None:
                writer.write(vis)

        if a.save:
            continue
        cv2.imshow("run_video", vis)
        wait = 0 if paused else max(1, int(1000 * ((idx + 1) / fps - (time.monotonic() - wall0))))
        key = cv2.waitKey(wait if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
            if not paused:
                wall0 = time.monotonic() - (idx + 1) / fps
        elif key == ord("d") and paused:
            step_once = True                                  # 일시정지 상태에서 한 프레임만 진행
        elif key == ord("o"):
            device("door", "open", t)
        elif key == ord("c"):
            device("door", "closed", t)
        elif key == ord("m"):
            device("car", "stopped" if dev["car"] == "moving" else "moving", t)
        elif key == ord("r"):
            device("operator", "reset", t)
        elif key == ord("s"):
            path = os.path.abspath(f"run_video_{idx:05d}.png")
            cv2.imwrite(path, vis)
            print("저장:", path)
    cap.release()
    if writer is not None:
        writer.release()
        print("결과 영상:", a.save)
    else:
        cv2.destroyAllWindows()
    print(f"\n최종 상태 {pl.state}")


if __name__ == "__main__":
    main()
