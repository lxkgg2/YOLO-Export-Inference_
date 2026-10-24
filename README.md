markdown
# YOLO 模型导出、推理与性能对比

## 1. 项目概述

本项目基于第一次作业的 YOLO26n 装甲板四关键点模型（`best.pt`），完成以下工作：

- 将 PyTorch 权重 `.pt` 导出为 ONNX 格式；
- 使用 `.pt` 和 `.onnx` 分别对同一段装甲板视频进行推理；
- 对比两种格式的模型大小、推理耗时、检测数量与检测效果。

## 2. 模型信息

- 模型版本：YOLO26n-pose（Ultralytics 8.4.185）
- 类别：B1、B2、B3、R1、R2、R3（6 类）
- 关键点：每个目标 4 个二维关键点（装甲板四角）
- 导出格式：ONNX（opset 18，端到端输出 `nms=False`）
- 输入尺寸：640×640

## 3. 运行环境

- Python 3.11.17
- PyTorch 2.11.0+cu128
- Ultralytics 8.4.185
- ONNX Runtime GPU 1.26.0（CUDA 12.8）
- OpenCV、NumPy
- 硬件：AMD Ryzen 5 5600 + NVIDIA GPU（CUDA 12.8）

## 4. 模型准备

将课程提供的 `best.pt` 放到 `weights/` 目录下。导出 ONNX：

```bash
python export.py
5. 依赖安装
bash
conda create -n yolo python=3.11
conda activate yolo
pip install -r requirements.txt
如使用 GPU 推理，需安装 onnxruntime-gpu 并确保 CUDA 12.x / cuDNN 9 可用。本项目中，inference_onnx.py 在 import onnxruntime 之前先 import torch，以加载 PyTorch 自带的 CUDA 运行库，解决 cublasLt64_12.dll 找不到的问题。

6. 运行命令
.pt 推理
bash
python inference_pt.py --model weights/best.pt --source videos/armor_raw.mp4 --output results/pt_result.mp4
.onnx 推理
bash
python inference_onnx.py --model weights/best.onnx --source videos/armor_raw.mp4 --output results/onnx_result_gpu.mp4
结果视频保存在 results/ 目录下。

7. 测试视频说明
拍摄地点：水建学院 D414 实验室

时长：约 18 秒（1072 帧，1920×1080，59.07 FPS）

场景：手持设备拍摄装甲板，包含不同距离与角度变化

原始视频 videos/armor_raw.mp4 未提交到 Git，运行前需自行放置。

8. 结果对比
对比项目	原始 .pt 模型	导出 .onnx 模型
模型格式	PyTorch .pt	ONNX
模型文件大小	5.6 MB	9.7 MB
推理耗时	21.59 ms/帧	10.18 ms/帧
总检测目标数	1283	1306
含目标帧数	1058	848
推理设备	GPU (CUDA 12.8)	GPU (CUDA 12.8)
耗时统计方式：跳过前 5 帧预热，对其余帧取平均。两者使用同一视频、同一输入尺寸 640×640、同一置信度阈值 0.25。

检测效果截图
.pt 结果：results/pt_screenshot.png

.onnx 结果：results/onnx_screenshot.png

（此处插入两张截图）

9. 问题与分析
9.1 ONNX 导出后的输出格式变化
首次导出未加参数，输出为 (1, 18, 8400) 的原始候选格式，需要外部 NMS。为简化推理流程，重新导出时指定 nms=False，使用 one-to-one 输出头，输出变为 (1, 300, 14)，模型内部已完成去重，推理代码无需手写 NMS。

9.2 .pt 与 .onnx 检测数量差异
.pt 含目标帧数 1058，.onnx 为 848，总检测数分别为 1283 和 1306。差异主要来自后处理机制不同：.pt 使用 one-to-many 头 + 外部 NMS，对小目标、低置信度目标更敏感；ONNX 端到端版使用 one-to-one 头，去重更严格，部分低置信度目标被过滤。两者检测效果整体一致，关键点位置准确。

9.3 GPU 推理环境问题
直接安装 onnxruntime-gpu 后，初始化 CUDA provider 时报错 cublasLt64_12.dll 缺失。原因是 PyTorch 自带的 CUDA 库不在 onnxruntime 的 DLL 搜索路径中，且 os.add_dll_directory 会被 onnxruntime 内部覆盖。解决方法是在 import onnxruntime 之前先 import torch，使 CUDA 运行库加载进当前进程，onnxruntime 随后可正常使用 GPU。最终推理耗时从 CPU 的 26.38 ms/帧降至 GPU 的 10.18 ms/帧。

10. 结论
ONNX 导出后模型文件变大（5.6 MB → 9.7 MB），因为 ONNX 存储的是未压缩计算图与权重。

在相同 GPU 硬件下，ONNX 推理速度约为 .pt 的 2.1 倍（10.18 ms vs 21.59 ms），主要得益于 ONNX Runtime 的图优化和端到端输出减少了后处理开销。

端到端导出（nms=False）简化了部署流程，但检测策略更保守，含目标帧数少于 .pt。

两种格式检测效果整体一致，关键点均来自模型预测，位置准确。

text

## 做完告诉我

1. 三个文件建好了没；
2. `README.md` 里第 8 节的两张截图有没有插入（把 `（此处插入两张截图）` 换成 `![](results/pt_screenshot.png)` 这种 Markdown 图片语法）；
3. 截图文件名是不是 `pt_screenshot.png` 和 `onnx_screenshot.png`，如果不一样告诉我实际名字，我帮你改 README 里的路径。

文件齐了之后，下一步就是 Git 提交和建仓库。