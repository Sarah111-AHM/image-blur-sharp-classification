# image-blur-sharp-classification
lightweight and interpretable approach for binary image blur detection using handcrafted spatial and frequency domain features processed by a fully connected multilayer perceptron neural network this method provides efficient realtime classification without relying on deep convolutional architectures
---
# Image Blur vs Sharp Classification

![Python Version](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-experimental-yellow)

---

## Table of Contents
1. [Description](#description)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Implementation](#implementation)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [License](#license)

---

## Description
a lightweight and interpretable approach for binary image blur detection using handcrafted spatial and frequency domain features processed by a fully connected multilayer perceptron neural network this method provides efficient realtime classification without relying on deep convolutional architectures

This project focuses on extracting meaningful features from images instead of raw pixels, providing a **fast, efficient, and interpretable solution** for blur detection suitable for low-resource and real-time applications.

![demo](https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif)

---

## Features
- Detects **blurred vs sharp images** accurately using handcrafted features
- Spatial domain features: Laplacian variance, edge density, gradient magnitude
- Frequency domain features: FFT energy, high-frequency ratio
- Lightweight **Multilayer Perceptron (MLP) neural network**
- Binary classification using **Sigmoid activation** and **Binary Cross Entropy loss**
- Optimized with **Adam optimizer** for fast convergence
- Real-time and low-resource suitable

---

## Dataset
Publicly available datasets used for training and testing:

- LIVE Image Quality Assessment Dataset  
- KADID-10K  
- Kaggle Blur Image Datasets

> Note: Sample images are included in `data/sample_images/`, large datasets are not uploaded.

---

## Model Architecture
- **Input Layer:** Feature vector (5–50 features)  
- **Hidden Layers:** 1–3 fully connected layers with ReLU activation  
- **Output Layer:** Single neuron with Sigmoid activation (binary classification)  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam or SGD  

![architecture](https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif)

---

## Implementation
- Feature extraction: `features/feature_extraction.py`  
- Model training: `model/mlp_classifier.py`  
- Experiments: `notebooks/experiments.ipynb`  
- Results and visualizations stored in `results/`  

```python
# Example: Training the MLP classifier
from mlp_classifier import train_model
train_model(features, labels)
