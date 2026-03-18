# YOLOv8 webcam facial expression recognition

This project trains a YOLOv8 classification model on the dataset under `G:\731\dataset`,
then uses a webcam for real-time facial expression recognition.

## Python version note

PyTorch on Windows currently supports Python `3.9-3.12` for the CUDA path. On this machine,
the recommended choice is Python `3.11`, otherwise training will fall back to CPU only.

## Project files

- `prepare_dataset.py`: rebuilds the dataset into YOLOv8 classification format.
- `train_emotion_yolov8.py`: trains a YOLOv8 classification model.
- `realtime_emotion_webcam.py`: runs real-time webcam inference.
- `.vscode/`: VS Code interpreter, tasks, and launch configs.

## Important dataset note

`labels.csv` contains the effective label for each image. The original folder names are not
fully reliable, so `prepare_dataset.py` uses the `label` column as ground truth.

## Recommended workflow

1. Open `G:\731` in VS Code.
2. Create a virtual environment and install dependencies.
3. Prepare the dataset split.
4. Train the YOLOv8 classification model.
5. Run the webcam program.

## Commands

```powershell
cd G:\731
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe prepare_dataset.py --overwrite
.\.venv311\Scripts\python.exe train_emotion_yolov8.py --epochs 30 --batch 64 --device 0
.\.venv311\Scripts\python.exe realtime_emotion_webcam.py --device 0
```

## Optional settings

- `prepare_dataset.py --min-confidence 0.8`: keep only higher-confidence labels.
- `train_emotion_yolov8.py --model yolov8s-cls.pt`: switch to a larger classifier.
- `realtime_emotion_webcam.py --skip-frames 1`: improve FPS on slower machines.

## Output locations

- Prepared dataset: `G:\731\emotion_cls_data`
- Training runs: `G:\731\runs`
- Best model: `G:\731\runs\emotion_yolov8n_cls\weights\best.pt`
