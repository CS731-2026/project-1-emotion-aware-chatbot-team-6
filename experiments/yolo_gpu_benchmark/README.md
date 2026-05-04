# YOLO Face Detection GPU Benchmark

Empirical comparison of YOLOv8, YOLOv11, and YOLOv12 face detectors on real driving footage, conducted on NVIDIA Tesla T4 GPU (Google Colab).

## Methodology

- **Dataset**: 30-second real driving video (Pexels, free license)
- **Frame count**: 150 frames per model
- **Hardware**: NVIDIA Tesla T4 GPU (16 GB VRAM, Google Colab)
- **Models**: yolov8n-face, yolov11n-face, yolov12n-face from [akanametov/yolo-face](https://github.com/akanametov/yolo-face)
- **Warm-up**: 5 frames per model excluded from timing
- **Synchronization**: `torch.cuda.synchronize()` used for accurate GPU timing

## Results

| Metric              | YOLOv8n-face | YOLOv11n-face | YOLOv12n-face |
|---------------------|--------------|---------------|---------------|
| Avg Inference (ms)  | **9.94**     | 11.27         | 20.57         |
| Min Inference (ms)  | 8.96         | 10.00         | 14.51         |
| Max Inference (ms)  | 16.50        | 19.20         | 35.92         |
| Avg FPS             | **100.62**   | 88.73         | 48.61         |
| Detection Rate      | 100%         | 100%          | 100%          |

## Findings

1. All three models achieved **100% face detection rate** on driving footage — accuracy is equivalent.
2. **YOLOv8n-face is fastest on GPU** at 9.94 ms average (100+ FPS), benefiting most from CUDA parallelism due to its pure CNN architecture.
3. **YOLOv12n-face is slowest** despite being newest, due to attention-based architecture overhead even on GPU.
4. Real-time performance (>30 FPS) is achievable for all three models on T4 GPU.

## Selection for DriveSense

Given that all three models meet real-time requirements and detection accuracy is equivalent, we select **YOLOv11n-face** as the production model based on its balanced performance across deployment scenarios.

## Files

| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `colab_gpu_compare.py`        | Benchmark script for Colab T4 runtime    |
| `summary_gpu.csv`             | Aggregate results table                  |
| `results_yolov8_gpu.csv`      | Per-frame inference times (YOLOv8)       |
| `results_yolov11_gpu.csv`     | Per-frame inference times (YOLOv11)      |
| `results_yolov12_gpu.csv`     | Per-frame inference times (YOLOv12)      |
| `compare_v8_gpu.jpg`          | Same-frame detection visualization (v8)  |
| `compare_v11_gpu.jpg`         | Same-frame detection visualization (v11) |
| `compare_v12_gpu.jpg`         | Same-frame detection visualization (v12) |

## How to Reproduce

1. Open [Google Colab](https://colab.research.google.com/) with **T4 GPU** runtime
2. Download model weights from [akanametov/yolo-face/releases](https://github.com/akanametov/yolo-face/releases):
   - `yolov8n-face.pt`
   - `yolov11n-face.pt`
   - `yolov12n-face.pt`
3. Upload weights and a driving video named `test_video.mp4` to Colab
4. Run `colab_gpu_compare.py`
