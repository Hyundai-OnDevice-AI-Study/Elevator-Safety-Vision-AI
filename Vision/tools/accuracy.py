"""[Pi에서 실행] 비전 정확도 측정 — 설계 문서 6.4 '정확도 보고'.

기준 모델(첫 번째 --models)과 비교해서 모델별로 아래 두 가지를 출력한다.
  1. 사람 검출률: 사람이 1명 이상 잡힌 프레임 비율
  2. 키포인트 차이: 기준 모델 대비 평균 픽셀 차이 (둘 다 conf >= 0.5 인 관절만)

  python tools/accuracy.py --models models/yolo11n-pose_320.onnx models/yolo11n-pose_320_int8_static.onnx \
                           --videos videos/leash videos/scarf --step 5

  # PC 에서 PyTorch(.pt)를 기준으로 ONNX 변환 손실 확인 (ultralytics 필요)
  python tools/accuracy.py --pt yolo11n-pose.pt --models models/yolo11n-pose_320.onnx --videos videos/leash

--step 5 : 5프레임마다 1장씩만 검사 (Pi 에서 시간 절약)
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from door_vision.pose_estimator import PoseEstimator  # noqa: E402


def list_videos(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.mp4"))) if os.path.isdir(p) else [p]
    return out


def read_frames(video, step):
    cap = cv2.VideoCapture(video)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield f
        i += 1
    cap.release()


def top_kpts_onnx(est, frame):
    """가장 점수 높은 사람의 (17, 3) 정규화 키포인트, 없으면 None."""
    r = est(frame)
    if not r:
        return None
    return max(r, key=lambda p: p.score).kpts


def make_pt_fn(pt_path, imgsz):
    from ultralytics import YOLO
    model = YOLO(pt_path)

    def fn(frame):
        r = model(frame, imgsz=imgsz, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        xyn = r.keypoints.xyn[i].cpu().numpy()
        conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
        return np.concatenate([xyn, conf[:, None]], axis=1)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--videos", nargs="+", default=["videos"])
    ap.add_argument("--pt", default="", help="PyTorch 기준 모델 (PC 전용)")
    ap.add_argument("--pt-imgsz", type=int, default=320)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--kpt-conf", type=float, default=0.5)
    a = ap.parse_args()

    names, fns = [], []
    if a.pt:
        names.append(os.path.basename(a.pt) + " (기준)")
        fns.append(make_pt_fn(a.pt, a.pt_imgsz))
    for m in a.models:
        est = PoseEstimator(m, threads=a.threads)
        names.append(os.path.basename(m) + ("" if fns else " (기준)"))
        fns.append(lambda f, e=est: top_kpts_onnx(e, f))

    videos = list_videos(a.videos)
    if not videos:
        sys.exit("영상이 없습니다: --videos 경로 확인")

    n_frames = 0
    detected = np.zeros(len(fns), int)
    diffs = [[] for _ in fns]
    for v in videos:
        for frame in read_frames(v, a.step):
            h, w = frame.shape[:2]
            res = [fn(frame) for fn in fns]
            n_frames += 1
            for j, r in enumerate(res):
                if r is None:
                    continue
                detected[j] += 1
                ref = res[0]
                if j == 0 or ref is None:
                    continue
                ok = (ref[:, 2] >= a.kpt_conf) & (r[:, 2] >= a.kpt_conf)
                if ok.any():
                    d = (ref[ok, :2] - r[ok, :2]) * np.array([w, h])
                    diffs[j].append(np.linalg.norm(d, axis=1).mean())
        print(f"  {os.path.basename(v)} 완료")

    print(f"\n영상 {len(videos)}개, 검사 프레임 {n_frames}장 (step={a.step})")
    print(f"{'model':44s} {'검출률':>7s} {'kpt차이(px)':>11s}")
    for j, name in enumerate(names):
        rate = detected[j] / max(n_frames, 1) * 100
        diff = "-" if j == 0 else (f"{np.mean(diffs[j]):.2f}" if diffs[j] else "n/a")
        print(f"{name:44s} {rate:6.1f}% {diff:>11s}")
    print("\nkpt차이는 원본 영상 픽셀 기준. 1280 폭 영상에 입력 320이면 입력 1px = 원본 4px 이라 5~8px 정도는 정상.")
    print("검출률은 사람이 계속 보이는 영상이면 100%에 가까워야 함.")


if __name__ == "__main__":
    main()
