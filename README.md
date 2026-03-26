# Face Detection + timm Classification Focus Monitor

This project now uses:

- YOLO face detection for face boxes
- `timm` image classification for emotion recognition
- `timm` image classification for open-eye / closed-eye recognition
- geometry-based eye boxes drawn inside each detected face

## Python version note

PyTorch on Windows currently supports Python `3.9-3.12` for the CUDA path. On this machine,
the recommended choice is Python `3.11`, otherwise training will fall back to CPU only.

## Project files

- `prepare_dataset.py`: prepares `emotion` and `eye` datasets into folder-based classification format.
- `train_emotion_yolov8.py`: older YOLOv8 emotion training script kept for comparison.
- `train_emotion_timm.py`: trains the timm emotion classifier benchmark.
- `train_eye_timm.py`: trains the timm eye-state classifier.
- `realtime_emotion_webcam.py`: runs real-time face detection with timm emotion and eye-state recognition.
- `summarize_timm_benchmark.py`: summarizes multiple emotion timm runs.
- `.vscode/`: VS Code interpreter and launch configs.

## Dataset layout

The source dataset is expected under `G:\731\dataset`:

- `emotion/`: folder-based classification dataset with `train`, `valid`, `test`
- `eye/`: classification dataset exported with `_classes.csv`
- `Affectnet-HQ/`: optional extra emotion dataset in `labels.csv + folders` format

`prepare_dataset.py` converts them into:

- `G:\731\prepared_datasets\emotion`
- `G:\731\prepared_datasets\eye`

The emotion task is standardized to 6 classes only:

- `anger`
- `disgust`
- `fear`
- `happy`
- `sad`
- `surprise`

The eye task uses 2 classes:

- `closed_eye`
- `open_eye`

If `Affectnet-HQ` exists, it is merged into the emotion training split automatically.
Rows with labels outside those 6 classes are skipped.

## Environment setup

```powershell
cd G:\731
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## Data preparation

```powershell
.\.venv311\Scripts\python.exe prepare_dataset.py --overwrite
```

## Training

Emotion benchmark model example:

```powershell
.\.venv311\Scripts\python.exe train_emotion_timm.py --model-key resnet50 --epochs 20 --batch-size 32 --img-size 224 --device cuda --overwrite
```

Eye model:

```powershell
.\.venv311\Scripts\python.exe train_eye_timm.py --device cuda:0 --overwrite
```

The eye training script is fixed to `EfficientNet-B0` and writes to:

- `G:\731\runs_timm\eye_efficientnet_b0\best_model.pth`
- `G:\731\runs_timm\eye_efficientnet_b0\metadata.json`

If you want to compare all five emotion models, run:

- `resnet50`
- `efficientnet_b0`
- `efficientnet_b3`
- `swin_tiny`
- `mobilenet_v2`

Then summarize them with:

```powershell
.\.venv311\Scripts\python.exe summarize_timm_benchmark.py --run-names resnet50 efficientnet_b0 efficientnet_b3 swin_tiny mobilenet_v2
```

## Real-time inference

```powershell
.\.venv311\Scripts\python.exe realtime_emotion_webcam.py --device 0 --window-width 1280 --window-height 720
```

`realtime_emotion_webcam.py` now resolves classifiers from `G:\731\runs_timm`, not from `G:\731\runs`.
It reads each run's `metadata.json` and `best_model.pth`.

You can also pass explicit checkpoints or run directories:

```powershell
.\.venv311\Scripts\python.exe realtime_emotion_webcam.py `
  --emotion-model G:\731\runs_timm\resnet50 `
  --eye-model G:\731\runs_timm\eye_efficientnet_b0 `
  --device 0
```

## Runtime behavior

- Green boxes show faces and emotion labels.
- Blue boxes show eyes and eye-state labels.
- When the primary face stays closed-eyed for 3 seconds or more, the overlay shows `Please stay focused`.
- Window size is adjustable with `--window-width` and `--window-height`.
- Capture resolution is adjustable with `--capture-width` and `--capture-height`.
