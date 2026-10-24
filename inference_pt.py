#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装甲板视频推理脚本（YOLO26n 四关键点模型，.pt 版本）

用于和 ONNX 导出模型做性能对比：
  - 相同视频、相同 imgsz、相同 conf
  - 统计预热后的平均推理耗时
  - 结果保存为 results/pt_result.mp4
"""

import argparse
import os
import sys
import time

import cv2
import torch
from ultralytics import YOLO

# 装甲板类别（与模型训练保持一致：B1 B2 B3 R1 R2 R3）
ARMOR_CLASSES = ["B1", "B2", "B3", "R1", "R2", "R3"]

# 每个目标预测的关键点数量（装甲板四角点）
NUM_KEYPOINTS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="装甲板视频推理与关键点可视化（.pt）")
    parser.add_argument("--model", type=str, default="weights/best.pt",
                        help="YOLO 模型权重路径（默认 weights/best.pt）")
    parser.add_argument("--source", type=str, default="videos/armor_raw.mp4",
                        help="输入视频路径")
    parser.add_argument("--output", type=str, default="results/pt_result.mp4",
                        help="推理结果视频保存路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值（默认 0.45）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理输入尺寸（默认 640）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备：cpu / 0 / 0,1 等；不填则自动选择 GPU 或 CPU")
    return parser.parse_args()


def resolve_device(device):
    """根据用户指定或硬件情况选择推理设备（CPU 或 GPU）。"""
    if device is not None:
        return device
    if torch.cuda.is_available():
        print("[Info] 检测到 CUDA，使用 GPU 推理。")
        return 0
    print("[Info] 未检测到 CUDA，使用 CPU 推理。")
    return "cpu"


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        sys.exit(f"[错误] 找不到模型权重: {args.model}")
    if not os.path.isfile(args.source):
        sys.exit(f"[错误] 找不到输入视频: {args.source}")

    print(f"[Info] 加载模型: {args.model}")
    model = YOLO(args.model)
    if not model.names or len(model.names) < len(ARMOR_CLASSES):
        print("[Info] 模型未内嵌类别名，使用手动指定类别列表。")
        model.names = {i: ARMOR_CLASSES[i] for i in range(len(ARMOR_CLASSES))}

    device = resolve_device(args.device)
    print(f"[Info] 推理设备: {device}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开视频: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Info] 视频信息: {fps:.2f} FPS, {width}x{height}, 共约 {total} 帧")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"[错误] 无法创建输出视频: {args.output}")

    frame_idx = 0
    det_frames = 0
    total_dets = 0
    infer_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=device, verbose=False)
        infer_times.append(time.perf_counter() - t0)

        r = results[0]
        boxes = r.boxes
        keypoints = r.keypoints

        if boxes is not None and len(boxes) > 0:
            det_frames += 1
            total_dets += len(boxes)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            kps_xy = None
            if keypoints is not None and keypoints.xy is not None:
                kps_xy = keypoints.xy.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                cls_name = model.names.get(clss[i], f"class{clss[i]}")
                conf = confs[i]

                color = (255, 0, 0) if cls_name.startswith("B") else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                top = max(0, y1 - th - 10)
                cv2.rectangle(frame, (x1, top), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if kps_xy is not None:
                    for j in range(min(NUM_KEYPOINTS, kps_xy.shape[1])):
                        kx, ky = int(kps_xy[i, j, 0]), int(kps_xy[i, j, 1])
                        cv2.circle(frame, (kx, ky), 6, (0, 255, 0), -1)
                        cv2.putText(frame, str(j + 1), (kx - 8, ky - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"[Info] 已处理 {frame_idx}/{total} 帧，含目标的帧数: {det_frames}")

    cap.release()
    writer.release()

    if infer_times:
        warm = infer_times[5:] if len(infer_times) > 5 else infer_times
        avg_ms = sum(warm) / len(warm) * 1000
    else:
        avg_ms = 0

    print(f"\n[完成] 推理结束。")
    print(f"  输入视频: {args.source}")
    print(f"  输出视频: {args.output}")
    print(f"  总帧数: {frame_idx}，含目标帧数: {det_frames}，总检测目标数: {total_dets}")
    print(f"  .pt 平均推理耗时: {avg_ms:.2f} ms/帧（去掉前 5 帧预热）")

    if det_frames == 0:
        print("[提醒] 全片未检测到任何目标，请检查权重、视频、推理参数和类别是否匹配。")


if __name__ == "__main__":
    main()