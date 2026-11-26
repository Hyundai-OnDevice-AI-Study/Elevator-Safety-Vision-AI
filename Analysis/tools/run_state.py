"""상태 머신 재생 — 녹화된 PoseFrame 과 문·카 이벤트를 시간 순서대로 넣어 상태 변화와 제어 명령을 본다.

  python tools/run_state.py --label cam03_leash_drag
  python tools/run_state.py --poses data/poses/x.jsonl --vision-config ../Vision/config/vision.json \\
         --events '[[0.0,"door","closed"],[1.0,"car","moving"]]'
  python tools/run_state.py --label cam03_leash_drag --reset-at 9.0      # 관리자 리셋까지
  python tools/run_state.py --label cam03_leash_drag --record out/cam03_state.jsonl

events 의 시각은 첫 PoseFrame 기준 초. kind: door(open/closing/closed) · car(stopped/moving) ·
fault(원인/clear) · operator(reset).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # Analysis/
for _p in (HERE, os.path.join(os.path.dirname(HERE), "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from door_analysis.pipeline import AnalysisPipeline  # noqa: E402
from door_common.door_geometry import DoorGeometry  # noqa: E402
from door_common.messages import ActionCommand, DeviceEvent, PoseFrame, StateChange  # noqa: E402
from door_common.transport import JsonlRecorder, read_jsonl  # noqa: E402


def run(frames, door, rules, events, quiet=False, recorder=None):
    """반환: (최종 상태, [(상대 시각, 메시지)])"""
    t0 = frames[0].ts
    evs = sorted(((t0 + t, DeviceEvent(t0 + t, k, v)) for t, k, v in events), key=lambda x: x[0])
    first_door = next((e.value for _, e in evs if e.kind == "door"), "closed")
    pl = AnalysisPipeline(door, rules, door_state=first_door, start_ts=t0)
    log = []

    def emit(outs):
        for m in outs:
            log.append((m.ts - t0, m))
            if recorder:
                recorder.publish(m)
            if quiet:
                continue
            if isinstance(m, StateChange):
                print(f"{m.ts - t0:6.2f}s  ── {m.prev} → {m.state}   ({m.reason})")
            elif isinstance(m, ActionCommand):
                extra = " ".join(f"{k}={v}" for k, v in m.params.items())
                print(f"{'':8s}   └ {m.target:6s} {m.action:15s} {extra}")

    k = 0
    for f in frames:
        while k < len(evs) and evs[k][0] <= f.ts:
            e = evs[k][1]
            if not quiet:
                print(f"{e.ts - t0:6.2f}s  [이벤트] {e.kind} = {e.value}")
            emit(pl.handle(e))
            k += 1
        emit(pl.handle(f))
    for _, e in evs[k:]:                                   # 영상이 끝난 뒤 이벤트 (리셋 등)
        if not quiet:
            print(f"{e.ts - t0:6.2f}s  [이벤트] {e.kind} = {e.value}")
        emit(pl.handle(PoseFrame(ts=e.ts, frame_id=-1)))     # 비전은 계속 '사람 없음'을 보낸다고 가정
        emit(pl.handle(e))
    return pl.state, log


def main():
    ap = argparse.ArgumentParser(description="상태 머신 재생")
    ap.add_argument("--label", help="config/scenarios/labels.json 의 name")
    ap.add_argument("--labels", default=os.path.join(HERE, "config", "scenarios", "labels.json"))
    ap.add_argument("--poses")
    ap.add_argument("--vision-config")
    ap.add_argument("--events", help='JSON: [[t, kind, value], ...]')
    ap.add_argument("--reset-at", type=float, help="이 시각(초)에 관리자 리셋 이벤트 추가")
    ap.add_argument("--rules", default=os.path.join(HERE, "config", "rules.json"))
    ap.add_argument("--record", help="StateChange·ActionCommand 를 jsonl 로 저장")
    a = ap.parse_args()

    events, expected = [], None
    if a.label:
        item = next((x for x in json.load(open(a.labels, encoding="utf-8")) if x["name"] == a.label), None)
        if item is None:
            ap.error(f"{a.label} 이 labels 에 없습니다")
        j = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))
        a.poses = a.poses or j(item["poses"])
        a.vision_config = a.vision_config or j(item["vision_config"])
        events = item.get("device_events", [])
        expected = item.get("expected_state")
    if a.events:
        events = json.loads(a.events)
    if a.reset_at is not None:
        events = events + [[a.reset_at, "operator", "reset"]]
    if not (a.poses and a.vision_config):
        ap.error("--label 또는 --poses 와 --vision-config 가 필요합니다")

    rules = json.load(open(a.rules, encoding="utf-8")) if os.path.exists(a.rules) else {}
    door = DoorGeometry.from_config(json.load(open(a.vision_config, encoding="utf-8")))
    frames = [f for f in read_jsonl(a.poses) if isinstance(f, PoseFrame)]
    rec = None
    if a.record:
        os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
        rec = JsonlRecorder(a.record)
    final, log = run(frames, door, rules, events, recorder=rec)
    if rec:
        rec.close()
    peak = max((m.state for _, m in log if isinstance(m, StateChange)),
               key=lambda s: ["DOOR_OPEN", "MONITORING", "WARNING", "FAULT", "DANGER_PRE", "DANGER_MOVING"].index(s),
               default=final)
    print(f"\n최종 상태 {final}, 가장 높은 상태 {peak}" + (f", 기대 {expected} → {'PASS' if peak == expected else 'FAIL'}"
                                                   if expected else ""))


if __name__ == "__main__":
    main()
