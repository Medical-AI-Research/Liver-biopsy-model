# Liver Biopsy Histopathology Classification using RegNet + Grad-CAM

## Overview
This project implements a **deep learning pipeline for multi-class classification of liver biopsy histopathology images** using a **RegNet (RegNetX-002)** backbone.  
It performs **training, validation, extensive evaluation, and explainability (Grad-CAM)** on five clinically relevant liver conditions.

The entire pipeline is designed to run in **Google Colab** with data stored on **Google Drive**.

---

## Dataset Structure
The dataset is expected to follow an `ImageFolder`-compatible structure:

Liver Biopsies/
├── Healthy/
├── Inflammation/
├── Steatosis/
├── Ballooning/
└── Fibrosis/


Each folder contains RGB histopathology images corresponding to that class.

---

## Classes
- Healthy  
- Inflammation  
- Steatosis  
- Ballooning  
- Fibrosis  

Total classes: **5**

---

## Model Architecture
- Backbone: **RegNetX-002 (ImageNet pretrained)**
- Framework: **PyTorch**
- Model source: `timm`
- Final classifier head is automatically adapted to 5 classes.

---

## Training Configuration
- Image size: **224 × 224**
- Batch size: **16**
- Epochs: **12**
- Optimizer: **AdamW**
- Learning rate: **2e-4**
- Weight decay: **1e-4**
- Scheduler: **Cosine Annealing**
- Loss function: **CrossEntropyLoss**
- Mixed precision: **Enabled (torch.cuda.amp)**
- Stratified train/validation split: **80% / 20%**
- Optional MixUp augmentation (disabled by default)

---

## Data Augmentation
**Training**
- Random resized crop
- Horizontal & vertical flip
- Color jitter
- Normalization (ImageNet mean/std)

**Validation**
- Resize + center crop
- Normalization only

---

## Evaluation Metrics
The pipeline computes **clinical-grade evaluation metrics**:

### Classification Metrics
- Accuracy (Train & Validation)
- Precision, Recall, F1-score (per class)
- Confusion Matrix

### Agreement & Reliability
- Cohen’s Kappa
- Matthews Correlation Coefficient (MCC)

### Probabilistic Metrics
- AUROC (per class)
- Macro-average AUROC (OvR)
- Precision–Recall curves
- Average Precision (AP)

### Optimization Metrics
- Cross-entropy loss
- Learning curves (loss & accuracy)

---

## Explainability (Grad-CAM)
- Automatic detection of the **last convolutional layer**
- Grad-CAM heatmaps overlaid on original biopsy images
- Supports:
  - Random validation samples
  - Stratified samples per class
  - Batch visualization (10+ images)
- Displays:
  - True class
  - Predicted class
  - Spatial attention map

This helps validate **pathology-relevant feature learning** rather than shortcut learning.

---
/MyDrive/regnet_checkpoints/

- Training history plots
- Confusion matrix heatmap
- ROC & Precision–Recall curves
- Grad-CAM visualizations

---

## How to Run
1. Upload dataset to Google Drive in the required folder structure.
2. Open the notebook in Google Colab.
3. Update:
 ```python
 DATASET_ROOT = "/content/drive/MyDrive/Liver Biopsies"

```
Run cells sequentially.

Best-performing model is saved automatically.

Key Strengths

Strong CNN backbone (RegNet)

Robust stratified evaluation

Medical-grade metrics beyond accuracy

Built-in interpretability (Grad-CAM)

Clean, modular training pipeline

Limitations

Single-center dataset assumption

No stain normalization

No cross-validation

No external test set

Image-level labels only (no pixel-level supervision)

Future Improvements

Stain normalization (Macenko / Reinhard)

Cross-validation

Class imbalance handling

Vision Transformers comparison

Weakly supervised localization

Deployment-ready inference script

Use Case

This pipeline is suitable for:

Medical AI research

Histopathology classification studies

Explainable AI demonstrations

Academic projects and publications
## Outputs
- Best model checkpoints saved to:
