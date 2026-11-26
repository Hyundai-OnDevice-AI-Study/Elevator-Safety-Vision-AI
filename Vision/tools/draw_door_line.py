"""영상 한 프레임에서 마우스로 문틈 선·버튼 패널·카 안쪽 점·다각형 영역을 찍어 vision*.json 의 "door" 에 저장.

  python tools/draw_door_line.py --video videos/sample.mp4 --config config/vision.json
  python tools/draw_door_line.py --video videos/leash/cam03_leash_drag.mp4 --config config/vision_cam03.json --frame 30

클릭 순서
  ① 문틈 위쪽 끝  ② 문틈 아래쪽 끝  ③ 버튼 패널 왼쪽 위  ④ 버튼 패널 오른쪽 아래  ⑤ 카 안쪽 아무 곳
  (버튼 패널이 화면에 없거나 다각형으로 그릴 거면 ③④ 대신 r 키로 건너뛰기)
  ⑥ 그다음부터는 선택: 다각형 영역 (v0.4 calibrate 에서 가져옴)
       하늘색 = exclude : 버튼 패널 (기울어져 보이거나 여러 개일 때)
       주황색 = ignore  : 거울·유리 등. 사람의 발이 이 안이면 판단에서 뺀다 (비친 사람 오탐 방지)

키
  n : 지금 다각형을 닫고 새 다각형 시작      m : 지금 다각형 종류 바꾸기 (exclude ↔ ignore)
  u : 마지막 점 되돌리기                    s : 저장 (+ 같은 이름의 _door.png 미리보기)
  q / ESC : 저장하지 않고 끝내기
영상마다 카메라 위치가 다르면 영상별 설정 파일을 따로 만든다.
"""
import argparse
import json
import os

import cv2
import numpy as np

STEPS = ["문틈 위쪽 끝", "문틈 아래쪽 끝", "버튼 패널 왼쪽 위", "버튼 패널 오른쪽 아래", "카 안쪽 아무 곳"]
COLORS = {"exclude": (255, 200, 0), "ignore": (0, 128, 255)}       # BGR: 하늘, 주황
HINT = ["click: gap TOP", "click: gap BOTTOM", "click: panel top-left (r: skip)", "click: panel bottom-right",
        "click: any point INSIDE the car"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config/vision.json")
    ap.add_argument("--frame", type=int, default=0, help="몇 번째 프레임에서 찍을지")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("프레임을 읽을 수 없습니다")
    h, w = img.shape[:2]
    pts = []                                 # ①~⑤ (정규화)
    polys, kinds = [[]], ["exclude"]         # ⑥ 다각형들

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        p = (round(x / w, 4), round(y / h, 4))
        if len(pts) < len(STEPS):
            pts.append(p)
            print(f"{STEPS[len(pts) - 1]}: {p}")
        else:
            polys[-1].append(p)

    P = lambda p: (int(p[0] * w), int(p[1] * h))
    win = "door  (n: new polygon  m: exclude/ignore  u: undo  s: save  q: quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    def render():
        vis = img.copy()
        if len(pts) >= 2:
            cv2.line(vis, P(pts[0]), P(pts[1]), (255, 0, 255), 2)
        if len(pts) >= 4 and pts[2] != pts[3]:
            cv2.rectangle(vis, P(pts[2]), P(pts[3]), COLORS["exclude"], 1)
        for i, p in enumerate(pts):
            if not (i in (2, 3) and p == (0, 0)):
                cv2.circle(vis, P(p), 5, (0, 255, 0) if i == 4 else (0, 0, 255), -1)
        for pg, kind in zip(polys, kinds):
            if len(pg) >= 2:
                cv2.polylines(vis, [np.array([P(p) for p in pg], np.int32)], len(pg) > 2, COLORS[kind], 2)
            for p in pg:
                cv2.circle(vis, P(p), 4, COLORS[kind], -1)
        return vis

    while True:
        vis = render()
        msg = HINT[len(pts)] if len(pts) < len(STEPS) else f"polygon ({kinds[-1]}): click points, n/m/u/s"
        cv2.putText(vis, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r") and len(pts) == 2:
            pts.extend([(0, 0), (0, 0)])
        elif k == ord("u"):
            if polys[-1]:
                polys[-1].pop()
            elif len(polys) > 1:
                polys.pop()
                kinds.pop()
            elif pts:
                pts.pop()
                if len(pts) == 3 and pts[2] == (0, 0):        # 건너뛴 패널은 한 번에 되돌림
                    del pts[2:]
        elif k == ord("n") and len(polys[-1]) >= 3:
            polys.append([])
            kinds.append("exclude")
        elif k == ord("m"):
            kinds[-1] = "ignore" if kinds[-1] == "exclude" else "exclude"
        elif k in (ord("q"), 27):
            print("\n저장하지 않았습니다")
            cv2.destroyAllWindows()
            return
        elif k == ord("s"):
            if len(pts) < len(STEPS):
                print(f"\n먼저 ①~⑤ 를 찍으세요 (지금 {len(pts)}/{len(STEPS)})")
                continue
            break
    cv2.destroyAllWindows()

    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    door = cfg.setdefault("door", {})
    door.update({"gap_p1": list(pts[0]), "gap_p2": list(pts[1]),
                 "button_panel": [pts[2][0], pts[2][1], pts[3][0], pts[3][1]],
                 "inside_point": list(pts[4])})
    for kind in ("exclude", "ignore"):
        door[kind] = [[list(p) for p in pg] for pg, kd in zip(polys, kinds) if kd == kind and len(pg) >= 3]
    with open(a.config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    preview = os.path.splitext(a.config)[0] + "_door.png"
    cv2.imwrite(preview, render())          # 마지막 클릭까지 반영한 그림
    print(f"\n저장 완료: {a.config}  (미리보기 {preview})")
    print(f"  exclude {len(door['exclude'])}개, ignore {len(door['ignore'])}개")


if __name__ == "__main__":
    main()
