"""[Pi에서 실행] 모델 여러 개를 같은 조건으로 비교: 순수 추론 시간 · end-to-end 시간 · 포즈 정확도.

  python tools/compare_models.py --models models/yolo11n-pose_320_raw.onnx models/yolo11n-pose_320.onnx \
         models/yolo11n-pose_320_int8_static.onnx --videos videos/leash videos/scarf

첫 번째 모델이 정확도의 '기준'이다 (보통 변환만 한 FP32 모델).

  순수추론   : 모델 실행(session.run)만. 임의 입력, 영상·전처리 없음
  end-to-end : 프레임 1장 → 전처리(letterbox) + 추론 + 후처리(NMS, 좌표 복원)까지. 영상 디코딩은 제외
  검출률     : 사람이 1명 이상 잡힌 프레임 비율
  kpt conf   : 가장 점수 높은 사람의 관절 17개 신뢰도 평균
  kpt 차이   : 기준 모델과의 관절 위치 차이 (원본 영상 픽셀, 둘 다 conf >= 0.5 인 관절만). 평균 / p95
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from door_vision.pose_estimator import PoseEstimator  # noqa: E402


def load_frames(paths, step, limit):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.mp4"), recursive=True)) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        cap, i = cv2.VideoCapture(f), 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                out.append(fr)
            i += 1
        cap.release()
    if limit and len(out) > limit:                       # 고르게 줄이기 (Pi 메모리 절약)
        out = [out[i] for i in np.linspace(0, len(out) - 1, limit).astype(int)]
    return out, len(files)


def pure_infer(path, threads, n, warmup=10):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    x = np.random.rand(*i.shape).astype(np.float32)
    for _ in range(warmup):
        s.run(None, {i.name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        s.run(None, {i.name: x})
        t.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--videos", nargs="+", default=["videos"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=200, help="순수 추론 반복 횟수")
    ap.add_argument("--step", type=int, default=5, help="영상에서 몇 프레임마다 1장을 쓸지")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--kpt-conf", type=float, default=0.5)
    a = ap.parse_args()

    frames, n_videos = load_frames(a.videos, a.step, a.max_frames)
    if not frames:
        sys.exit("영상 프레임이 없습니다: --videos 경로 확인")
    print(f"영상 {n_videos}개에서 프레임 {len(frames)}장 사용, 스레드 {a.threads}\n")

    rows, ref = [], None
    for m in a.models:
        mean, p95 = pure_infer(m, a.threads, a.n)
        est = PoseEstimator(m, threads=a.threads)
        for f in frames[:5]:
            est(f)
        e2e, det, kconf, kpts = [], 0, [], []
        for f in frames:
            t0 = time.perf_counter()
            r = est(f)
            e2e.append((time.perf_counter() - t0) * 1e3)
            if r:
                best = max(r, key=lambda p: p.score)
                det += 1
                kconf.append(float(best.kpts[:, 2].mean()))
                kpts.append(best.kpts)
            else:
                kpts.append(None)
        diff = []
        if ref is not None:
            for f, k0, k1 in zip(frames, ref, kpts):
                if k0 is None or k1 is None:
                    continue
                ok = (k0[:, 2] >= a.kpt_conf) & (k1[:, 2] >= a.kpt_conf)
                if ok.any():
                    h, w = f.shape[:2]
                    diff += list(np.linalg.norm((k0[ok, :2] - k1[ok, :2]) * np.array([w, h]), axis=1))
        else:
            ref = kpts
        rows.append((os.path.basename(m), os.path.getsize(m) / 1e6, mean, p95, float(np.mean(e2e)),
                     1000 / float(np.mean(e2e)), det / len(frames) * 100, float(np.mean(kconf)) if kconf else 0.0,
                     (float(np.mean(diff)), float(np.percentile(diff, 95))) if diff else None))
        print(f"  {os.path.basename(m)} 완료")

    print(f"\n{'model':40s} {'MB':>5s} {'순수추론':>8s} {'p95':>7s} {'e2e':>7s} {'e2eFPS':>6s} {'검출률':>6s} "
          f"{'kptconf':>7s} {'kpt차이 평균/p95(px)':>20s}")
    for name, mb, mean, p95, e2e, fps, det, kc, d in rows:
        ds = "기준" if d is None and name == rows[0][0] else ("n/a" if d is None else f"{d[0]:.1f} / {d[1]:.1f}")
        print(f"{name:40s} {mb:5.1f} {mean:7.1f}ms {p95:6.1f}ms {e2e:6.1f}ms {fps:6.1f} {det:5.1f}% {kc:7.3f} {ds:>20s}")
    print("\nkpt차이는 원본 영상 픽셀 기준 (1280 폭 영상에 입력 320이면 입력 1px = 원본 4px).")


if __name__ == "__main__":
    main()
