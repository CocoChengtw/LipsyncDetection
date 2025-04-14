# Lip-Sync Based Deepfake Detection with Gemini-Guided Preprocessing
This repo contains an extended preprocessing pipeline built on top of [LipForensics](https://arxiv.org/abs/2012.07657), adapted for a custom lipsync demo dataset and includes Gemini-based preprocessing selection.

---

## Setup
### Install dependencies
```bash
pip install -r requirements.txt
```

> Tested with Python 3.10
> Suggest re-download model weights from [LipForensics](https://drive.google.com/file/d/1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7/view)

---

## Directory Structure
```
preprocessing/
├── preprocessing.py         # Frame alignment and mouth crop
├── detector.py              # Batch processing videos with Gemini-labeled time ranges
├── gemini_preprocess.ipynb  # Gemini-based segment filtering logic
evaluate.py                  # Model evaluation (same as LipForensics)
```

---

## Dataset Overview

We experimented with a small lip-sync-specific dataset (DemoDataset), consisting of:

- `safe_videos/`: Real videos without manipulation
- `scam_videos/`: Suspected deepfake videos
- `result_by_safe_video_exp.json` & `result_by_scam_video_exp.json`: Gemini-generated time windows for evaluation

---

## Preprocessing Experiments
We explored multiple preprocessing strategies:
### First Attempt (Entire video)

Included full-length videos, often unsuitable for lip-sync detection.

Issues:

- Multiple people in frame
- Fast motion or face not visible
- Mouth covered or blurry
- Video cuts and scene changes

| Confusion Matrix    | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| `[[238 251][4 38]]` | 52.0%    | 13.15%    | 90.48% | 22.93%   |

### Target: Sample Real (Human-selected)
Manually picked good-quality lipsync segments

| Confusion Matrix | Accuracy | Precision | Recall | F1-Score |
| ---------------- | -------- | --------- | ------ | -------- |
| `[[45 5][3 39]]` | 91.3%    | 88.64%    | 92.86% | 90.68%   |

> Limitation: Manual selection not scalable.

---

## Gemini Preprocess Experiments
In order to reach the target result: we used Gemini to automate lip-sync-friendly segment selection:

| Experiment                    | Accuracy | Precision | Recall | F1-Score | Notes                          |
| ----------------------------- | -------- | --------- | ------ | -------- | ------------------------------ |
| After Gemini Preprocess exp 1 | 64.9%    | 20.5%     | 85%    | 32.9%    | Still noisy segments           |
| After Gemini Preprocess exp 2 | 74.0%    | 50.0%     | 69.2%  | 58.0%    | Better filtering rules applied |
| After Gemini Preprocess exp 3 | 75.3%    | 53.3%     | 51.5%  | 57.1%    | Balanced precision & recall    |

> While Gemini helps reduce noise, performance is still behind manually selected data. Future directions include combining with motion filters or audio alignment techniques.

---

## Run the Pipeline
### Step 1: Preprocess videos
```bash
python preprocessing/detector.py
```

### Step 2: Evaluate model
```bash
python evaluate.py --dataset DemoDataset --weights_forgery ./models/weights/lipforensics_ff.pth
```

---

## Future Work
- Improve Gemini rules with lip-motion analysis
- Add audio-visual alignment filtering
- Test on larger, more diverse datasets
- Fine-tune the model using Gemini-preprocessed data

---
## Reference
Original Paper:\
**Lips Don't Lie: A Generalisable and Robust Approach To Face Forgery Detection**\
Haliassos et al., CVPR 2021\
[Paper](https://arxiv.org/abs/2012.07657)
[FaceDetector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)