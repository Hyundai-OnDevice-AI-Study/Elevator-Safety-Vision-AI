"""[PC에서 실행] Ultralytics YOLO-pose / detect 모델을 ONNX 로 변환.

  pip install ultralytics onnx onnxslim
  python tools/export_onnx.py --model yolo11n-pose.pt --imgsz 320 416 640
  python tools/export_onnx.py --model yolo11n.pt --imgsz 320          # 2차: 반려동물(cat/dog) 탐지용
  python tools/export_onnx.py --model yolo26n-pose.pt --imgsz 320 --nms-free   # YOLO26: NMS 없는 출력 (1, 300, 57)

YOLO11 기본 출력은 (1, 56, N) 이라 NMS 를 코드에서 하고, YOLO26 --nms-free 출력은 (1, 300, 57) 이라 NMS 가 필요 없다.
door_vision/pose_estimator.py 는 두 형식을 모두 읽는다. 둘 다 변환해서 benchmark.py 로 비교해 볼 것.

결과 파일을 Pi 의 models/ 폴더로 복사한다. 파일명에 입력 크기를 붙여 둔다 (예: yolo11n-pose_320.onnx).

팁
  - Pi 4B 에서는 n(nano) 모델 + 입력 320~416 이 현실적이다. 640 은 정확하지만 느리다.
  - imgsz 는 32의 배수여야 한다.
  - dynamic=False(고정 입력)가 onnxruntime 최적화에 유리하다.
  - opset 은 onnxruntime 버전과 맞아야 한다. 오류가 나면 --opset 12 로 낮춰 본다.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[320])
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--nms-free", action="store_true", help="YOLO26/YOLOv10 의 NMS 없는 head 로 내보내기 (nms=False)")
    ap.add_argument("--no-slim", action="store_true",
                    help="onnxslim 정리 없이 변환만 (비교용). 파일명 끝에 _raw 가 붙는다")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    from ultralytics import YOLO  # PC 에만 설치

    Path(a.out).mkdir(exist_ok=True)
    for sz in a.imgsz:
        m = YOLO(a.model)
        kw = dict(format="onnx", imgsz=sz, dynamic=False, simplify=not a.no_slim)
        if a.opset:
            kw["opset"] = a.opset
        if a.nms_free:
            kw["nms"] = False
        path = Path(m.export(**kw))
        dst = Path(a.out) / f"{Path(a.model).stem}_{sz}{'_raw' if a.no_slim else ''}.onnx"
        shutil.move(str(path), dst)
        print("저장:", dst)

        import onnxruntime as ort
        s = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
        print("  입력:", [(i.name, i.shape) for i in s.get_inputs()])
        print("  출력:", [(o.name, o.shape) for o in s.get_outputs()])


if __name__ == "__main__":
    main()
