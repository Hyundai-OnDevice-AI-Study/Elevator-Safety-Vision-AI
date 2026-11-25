"""영상 첫 프레임에서 마우스로 문틈 선·버튼 패널·카 안쪽 점을 찍어 vision.json 에 저장.

  python tools/draw_door_line.py --video videos/sample.mp4 --config config/vision.json

순서: ① 문틈 위쪽 끝 ② 문틈 아래쪽 끝 ③ 버튼 패널 왼쪽 위 ④ 버튼 패널 오른쪽 아래 ⑤ 카 안쪽 아무 곳
(버튼 패널이 화면에 없으면 ③④ 대신 r 키로 건너뛰기)  s: 저장, q: 취소
영상마다 카메라 위치가 다르면 영상별 설정 파일을 따로 만든다.
"""
import argparse
import json

import cv2

STEPS = ["문틈 위쪽 끝", "문틈 아래쪽 끝", "버튼 패널 왼쪽 위", "버튼 패널 오른쪽 아래", "카 안쪽 아무 곳"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config/vision.json")
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, img = cap.read()
    if not ok:
        raise SystemExit("프레임을 읽을 수 없습니다")
    h, w = img.shape[:2]
    pts = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(STEPS):
            pts.append((round(x / w, 4), round(y / h, 4)))
            print(f"{STEPS[len(pts) - 1]}: {pts[-1]}")

    cv2.namedWindow("door")
    cv2.setMouseCallback("door", on_mouse)
    while True:
        vis = img.copy()
        for p in pts:
            cv2.circle(vis, (int(p[0] * w), int(p[1] * h)), 5, (0, 0, 255), -1)
        if len(pts) >= 2:
            cv2.line(vis, (int(pts[0][0] * w), int(pts[0][1] * h)), (int(pts[1][0] * w), int(pts[1][1] * h)),
                     (255, 0, 255), 2)
        msg = STEPS[len(pts)] if len(pts) < len(STEPS) else "s: 저장"
        cv2.putText(vis, f"click: {len(pts) + 1}/{len(STEPS)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        print("\r다음:", msg, end="")
        cv2.imshow("door", vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r") and len(pts) == 2:
            pts.extend([(0, 0), (0, 0)])
        if k == ord("q"):
            return
        if k == ord("s") and len(pts) == len(STEPS):
            break
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("door", {})
    cfg["door"].update({"gap_p1": list(pts[0]), "gap_p2": list(pts[1]),
                        "button_panel": [pts[2][0], pts[2][1], pts[3][0], pts[3][1]],
                        "inside_point": list(pts[4])})
    with open(a.config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("\n저장 완료:", a.config)


if __name__ == "__main__":
    main()
