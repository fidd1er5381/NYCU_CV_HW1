# NYCU_CV_HW1

StudentID: 313553023  
Name: 褚敏匡

# NYCU Computer Vision 2025 Spring - Homework 1

## Introduction

This repository contains the implementation of a custom ResNetRS50 model for image classification, developed as part of the NYCU Computer Vision course in Spring 2025.

## Prerequisites

### Software Requirements
- Python 3.8+
- CUDA 11.3+
- PyTorch 1.10+
- torchvision
- matplotlib
- seaborn
- pandas
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fidd1er5381/NYCU_CV_HW1.git
cd NYCU_CV_HW1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Organize your dataset in the following structure:
```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    └── images/
```

## Train and Predict

To start train and predict, run:
```bash
python run.py
```

## Performance Snapshot
![image](https://github.com/user-attachments/assets/088e1200-c5bd-46db-a1c4-b5c2ce47773e)

### Model Architecture
- Base Model: ResNetRS50
- Input Size: 224x224
- Squeeze-and-Excitation: Integrated
- Data Augmentation: RandAugment, ColorJitter, Random Erasing

### Training Configuration
- Optimizer: AdamW
- Learning Rate: 1e-4
- Weight Decay: 0.05
- Learning Rate Scheduler: Cosine Annealing
- Label Smoothing: 0.1

### Performance Metrics
- Training Accuracy: 94.3%
- Validation Accuracy: 89.2%

## Visualization

The training script generates:
- Training/Validation Loss Curves
- Training/Validation Accuracy Curves
- Confusion Matrix

## Outputs

- Best Model: `best_model_resnetrs.pth`
- Prediction Results: `output_resnetrs.xlsx`
- Training Curves: `training_curve.png`
- Confusion Matrix: `confusion_matrix.png`
