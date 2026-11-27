"""[PC에서 실행] ONNX 모델 INT8 양자화 (선택 사항).

  pip install onnxruntime onnx
  python tools/quantize_int8.py models/yolo11n-pose_320.onnx --calib videos/ --mode static

주의
  - 양자화하면 모델이 작아지고 빨라질 '수도' 있지만, Pi 4B(ARM) 에서는 속도 이득이 들쭉날쭉하고
    키포인트 정확도가 떨어지는 경우가 많다. 반드시 benchmark.py(속도)와 evaluate.py(정확도)로
    FP32 와 비교한 뒤 채택한다.
  - static 모드는 실제 영상 프레임(calibration)으로 값 범위를 측정하므로 dynamic 보다 정확하다.
"""
import argparse
import glob
import os

import cv2
import numpy as np


def frames_from(path: str, size: int, n: int = 100):
    # 하위 폴더(videos/leash, videos/scarf ...)까지 찾는다
    files = sorted(glob.glob(os.path.join(path, "**", "*.mp4"), recursive=True)) if os.path.isdir(path) else [path]
    if not files:
        raise SystemExit(f"캘리브레이션 영상이 없습니다: {path}")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from door_vision.onnx_backend import letterbox, to_blob
    out = []
    for f in files:
        cap = cv2.VideoCapture(f)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        for idx in np.linspace(0, total - 1, max(1, n // len(files))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok:
                out.append(to_blob(letterbox(fr, size)[0]))
    return out


def head_nodes(model_path: str):
    """출력에서 거꾸로 올라가며 마지막 Conv 이후의 노드 이름을 모은다 (양자화에서 제외할 후처리 부분).

    YOLO 출력은 박스(0~320) · 점수(0~1) · 키포인트가 한 텐서로 합쳐진다. 이 부분까지 8비트로 줄이면
    점수가 1.25 단위로만 표현돼서 score·keypoint conf 가 망가진다. 그래서 Conv 까지만 양자화한다.
    """
    import onnx
    g = onnx.load(model_path).graph
    producer = {o: n for n in g.node for o in n.output}
    todo, seen = [o.name for o in g.output], set()
    while todo:
        n = producer.get(todo.pop())
        if n is None or n.name in seen or n.op_type == "Conv":
            continue
        seen.add(n.name)
        todo += list(n.input)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["dynamic", "static"], default="static")
    ap.add_argument("--calib", help="캘리브레이션용 영상 파일 또는 폴더 (static)")
    ap.add_argument("--size", type=int, default=0, help="입력 크기 (기본: 모델에서 읽음)")
    a = ap.parse_args()
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = a.model.replace(".onnx", "_pre.onnx")
    quant_pre_process(a.model, pre)
    out = a.model.replace(".onnx", f"_int8_{a.mode}.onnx")
    if a.mode == "dynamic":
        quantize_dynamic(pre, out, weight_type=QuantType.QUInt8)
    else:
        import onnxruntime as ort
        inp = ort.InferenceSession(pre, providers=["CPUExecutionProvider"]).get_inputs()[0]
        name = inp.name
        size = a.size or int(inp.shape[-1])
        data = frames_from(a.calib, size)
        exclude = head_nodes(pre)
        print(f"입력 {size}, 캘리브레이션 {len(data)}장, 양자화 제외 노드 {len(exclude)}개")

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter([{name: d} for d in data])

            def get_next(self):
                return next(self.it, None)

        quantize_static(pre, out, Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        nodes_to_exclude=exclude)
    print("저장:", out)


if __name__ == "__main__":
    main()
