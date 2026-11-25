"""[Pi에서 실행] 비전 속도 측정 — 설계 문서 '비전 팀: FPS 측정'.

  python tools/benchmark.py --models models/yolo11n-pose_320.onnx models/yolo11n-pose_416.onnx \
                            --video videos/sample.mp4 --threads 2 4 --n 100

출력: 모델·스레드별 전처리/추론/후처리 시간(평균, p95), 전체 FPS, CPU 온도.
Pi 4B 는 오래 돌리면 80°C 근처에서 클럭을 낮춘다(스로틀링). 방열판/팬 유무에 따라 FPS가 달라지므로
반드시 1~2분 이상 연속으로 돌려서 '나중 FPS'를 본다 (--n 1000 등).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from door_drag.vision.pose_estimator import PoseEstimator  # noqa: E402


def cpu_temp() -> str:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return f"{int(f.read()) / 1000:.1f}°C"
    except OSError:
        return "n/a"


def throttled() -> str:
    try:
        import subprocess
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except (OSError, FileNotFoundError):
        return "n/a"


def frames(video: str, n: int):
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        out = []
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            out.append(f)
        return out
    print("영상이 없어 무작위 이미지로 측정합니다 (탐지 결과는 의미 없음, 속도만 참고)")
    return [np.random.randint(0, 255, (480, 640, 3), np.uint8) for _ in range(min(n, 20))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--threads", type=int, nargs="+", default=[4])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    imgs = frames(a.video, a.n)
    print(f"{'model':38s} {'thr':>3s} {'pre':>6s} {'infer':>7s} {'p95':>7s} {'post':>6s} {'FPS':>6s} "
          f"{'persons':>7s} temp")
    for m in a.models:
        for th in a.threads:
            est = PoseEstimator(m, threads=th)
            for i in range(a.warmup):
                est(imgs[i % len(imgs)])
            pre, inf, post, npers = [], [], [], []
            t0 = time.perf_counter()
            for i in range(a.n):
                r = est(imgs[i % len(imgs)])
                pre.append(est.last_timing["pre_ms"])
                inf.append(est.last_timing["infer_ms"])
                post.append(est.last_timing["post_ms"])
                npers.append(len(r))
            fps = a.n / (time.perf_counter() - t0)
            print(f"{os.path.basename(m):38s} {th:3d} {np.mean(pre):6.1f} {np.mean(inf):7.1f} "
                  f"{np.percentile(inf, 95):7.1f} {np.mean(post):6.1f} {fps:6.1f} {np.mean(npers):7.2f} "
                  f"{cpu_temp()}")
    print("throttled:", throttled(), "(0x0 이 아니면 전원 부족/과열로 클럭이 떨어진 것)")


if __name__ == "__main__":
    main()
