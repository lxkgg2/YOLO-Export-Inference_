import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import onnxruntime as ort

ARMOR_CLASSES = ["B1", "B2", "B3", "R1", "R2", "R3"]
NUM_KEYPOINTS = 4
INPUT_SIZE = 640


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX 装甲板视频推理（GPU）")
    parser.add_argument("--model", type=str, default="weights/best.onnx")
    parser.add_argument("--source", type=str, default="videos/armor_raw.mp4")
    parser.add_argument("--output", type=str, default="results/onnx_result_gpu.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def preprocess(frame):
    img, scale, pad_x, pad_y = letterbox(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return np.ascontiguousarray(img), scale, pad_x, pad_y


def map_back(x, y, scale, pad_x, pad_y):
    return (x - pad_x) / scale, (y - pad_y) / scale


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"[错误] 找不到模型: {args.model}")
    if not os.path.isfile(args.source):
        sys.exit(f"[错误] 找不到视频: {args.source}")

    print(f"[Info] 加载 ONNX 模型: {args.model}")
    sess = ort.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"[Info] 实际使用的 providers: {sess.get_providers()}")
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开视频: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Info] 视频: {fps:.2f} FPS, {width}x{height}, 约 {total} 帧")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    frame_idx = 0
    det_frames = 0
    total_dets = 0
    infer_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        inp, scale, px, py = preprocess(frame)

        t0 = time.perf_counter()
        out = sess.run([out_name], {in_name: inp})[0][0]
        infer_times.append(time.perf_counter() - t0)

        frame_has_det = False
        for row in out:
            conf = float(row[4])
            if conf < args.conf:
                continue
            frame_has_det = True
            total_dets += 1

            x1, y1 = map_back(row[0], row[1], scale, px, py)
            x2, y2 = map_back(row[2], row[3], scale, px, py)
            cls_id = int(row[5])
            cls_name = ARMOR_CLASSES[cls_id] if 0 <= cls_id < len(ARMOR_CLASSES) else f"cls{cls_id}"
            color = (255, 0, 0) if cls_name.startswith("B") else (0, 0, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for k in range(NUM_KEYPOINTS):
                kx, ky = map_back(row[6 + k * 2], row[7 + k * 2], scale, px, py)
                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
                cv2.putText(frame, str(k + 1), (int(kx) - 8, int(ky) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if frame_has_det:
            det_frames += 1

        writer.write(frame)
        if frame_idx % 50 == 0:
            print(f"[Info] 已处理 {frame_idx}/{total} 帧，含目标帧: {det_frames}")

    cap.release()
    writer.release()

    if infer_times:
        warm = infer_times[5:] if len(infer_times) > 5 else infer_times
        avg_ms = sum(warm) / len(warm) * 1000
    else:
        avg_ms = 0

    print(f"\n[完成] 输出: {args.output}")
    print(f"  总帧数: {frame_idx}，含目标帧数: {det_frames}，总检测目标数: {total_dets}")
    print(f"  ONNX 平均推理耗时: {avg_ms:.2f} ms/帧（去掉前 5 帧预热）")


if __name__ == "__main__":
    main()