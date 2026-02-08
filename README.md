# U-Net Segmentation Experiments


이 프로젝트는 **U-Net 기반 이미지 segmentation**을 직접 구현하고,  
구조 요소(activation function, skip connection, epoch 수 등)가  
학습 성능에 어떤 영향을 주는지 실험·분석하기 위한 코드입니다.
---

## 📌 Project Goals

- U-Net 구조를 처음부터 구현하며 내부 동작 이해
- Skip Connection의 역할 실험 (ON / OFF)
- Activation Function 비교 (ReLU, LeakyReLU)
- Epoch 수 증가에 따른 성능 변화 관찰
- 수치(metric)뿐 아니라 **시각적 결과 중심 분석**

---

## 📁 Project Structure
```yaml
segmentation/
├─ data/
│ ├─ imgs/ # input images
│ └─ masks/ # ground-truth masks
│
├─ model.py # U-Net / No-Skip U-Net 정의
├─ dataset.py # Custom Dataset (image-mask pair)
├─ main.py # train / infer / visualization
│
├─ outputs/ # experiment outputs (gitignored)
│ └─ exp_xx/
│ ├─ checkpoints/
│ └─ predictions/
│
├─ results/ # representative results for GitHub
│ ├─ baseline/
│ ├─ no_skip/
│ └─ leakyrelu/
│
├─ configs/
│ └─ config.yaml
│
├─ requirements.txt
└─ README.md
```
---

## 📶 Dataset
> https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
- 이미지 수: 약 2667장
- 데이터형태:
    - `images/` 폴더 : 실제 RGB 이미지
    - `masks/` 폴더 : 각 이미지에 대응하는 이진 마스크 (사람 vs 배경)
- 목표: 사람 픽셀을 구별
---

## 🧠 Model Variants

### 1️⃣ Baseline U-Net
- Encoder–Decoder 구조
- Skip Connection 사용
- ReLU activation
- BCE + Dice Loss

### 2️⃣ No-Skip U-Net
- Skip Connection 제거
- 단순 Encoder–Decoder 구조
- 공간 정보 복원 성능 비교 목적

### 3️⃣ Activation Experiments
- ReLU vs LeakyReLU
- 초기 수렴 속도 및 안정성 비교

---

## 🧪 Training & Inference

### ▶ Train
```bash
python main.py --mode train
```

### ▶ Inference (결과 이미지 저장)
```bash
python main.py --mode show --index 0
```

### ▶ Dataset Visualization (GT 확인)
```bash
python main.py --mode show --index 0
```
---

## 📊 Evaluation Metrics

- Train Loss
  - BCEWithLogitsLoss + DiceLoss
  - 낮을수록 좋음
- Validation Dice Score
  - 0 ~ 1 범위
  - 1에 가까울수록 segmentation 성능 우수

>⚠️ Train loss 감소 ≠ 일반화 성능 향상 
>반드시 validation metric과 시각적 결과를 함께 확인

---

## 🖼️ Results (Qualitative Comparison)

Representative results are stored in results/ directory
to visually compare different experimental settings.

Input Image

Ground Truth Overlay

Prediction Overlay

> 실제 분석에서는 같은 입력 이미지에 대해
>서로 다른 모델 결과를 비교하는 방식으로 진행

---
## 🔍 Key Observations
- Skip Connection 제거 시 train loss는 더 낮아질 수 있으나,
  - 경계 복원 성능은 크게 저하됨
- Epoch 수를 늘린다고 항상 성능이 좋아지지 않음
  - 일정 시점 이후 과적합 발생
- Activation 변경 효과는 데이터 크기 및 BN 사용 여부에 따라 달라짐

---
## 🛠️ Environment

- Python 3.x
- PyTorch
- CUDA (optional)
- matplotlib, numpy, PIL

```bash
pip install -r requirements.txt
```