# Brain Tumor Detection with Deep Learning

A high-accuracy brain tumor detection system built using deep learning and transfer learning. This project uses VGG16 and TensorFlow to classify MRI scans into glioma, meningioma, pituitary tumors, or no tumor.

---

## Problem Statement

Brain tumors are a serious health concern, and early detection is crucial for effective treatment. Analyzing MRI scans manually is time-consuming and requires specialized expertise. This project automates the detection of brain tumors in MRI scans using deep learning, providing a fast and accurate classification to assist medical professionals.

---

## Features

- Preprocessed and augmented MRI dataset (2870+ images)
- Transfer learning with VGG16 base model
- Custom top layers for classification
- Data augmentation for robustness
- Streamlit web app for easy use
- Achieved 98% accuracy on test data
- Saved model for deployment

---

## Model Architecture

- Base Model: VGG16 (with weights from ImageNet)
- Optimizer: Adam (learning rate=0.0001)
- Loss: Sparse Categorical Crossentropy

---

## Performance

| Metric           | Value | 
|----------------|----------|
| Test Accuracy	   | 98%   | 
| Precision   | 0.98   |
| Recall | 0.98 | 
| F1-Score	   | 0.98   | 

---

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- NumPy

## Demo
You can access the live demo of the application by visiting the following link:
[View Demo](https://braintumordetectionjatinwig.streamlit.app/)
