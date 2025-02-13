# Real-Time Seizure Detection Web Application

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Deployment](#deployment)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Problem Statement
Seizures are sudden and uncontrolled electrical disturbances in the brain that can cause various symptoms, from mild confusion to loss of consciousness. Real-time seizure detection using EEG (Electroencephalography) signals is crucial for providing timely medical intervention. Traditional detection methods rely on manual analysis by neurologists, which is time-consuming and subjective.

This project aims to develop a real-time seizure detection system using deep learning techniques, allowing automated identification of seizure events from EEG signals. The system is deployed as a web application for ease of use and accessibility.

## Solution Approach
Our solution leverages deep learning models to analyze EEG data and detect seizures in real-time. The key steps involved are:
1. **Data Preprocessing:** Cleaning and normalizing EEG data for better model performance.
2. **Feature Engineering:** Extracting meaningful features to improve classification accuracy.
3. **Deep Learning Model:** Training a CNN-ViT hybrid model to classify seizure vs. non-seizure events.
4. **Visualization:** Using PyTorch hooks to highlight important EEG channels contributing to predictions.
5. **Web Application:** Deploying the model using FastAPI and integrating a user-friendly interface with Streamlit.
6. **Deployment:** Hosting the application for real-time inference and monitoring.

## Dataset
The dataset consists of EEG recordings from multiple patients, including seizure and non-seizure instances. Data is preprocessed to ensure uniform sampling and proper feature extraction.

## Feature Engineering
- Applied Fourier Transform for frequency domain analysis.
- Extracted statistical features such as mean, variance, skewness, and kurtosis.
- Used bandpass filtering to isolate relevant frequency bands (delta, theta, alpha, beta, gamma).
- Performed PCA to reduce dimensionality while preserving critical information.

## Model Architecture
- Used a **CNN-ViT hybrid** model for learning spatial and temporal features of EEG signals.
- CNN layers capture local patterns, while Vision Transformers focus on long-range dependencies.
- Applied dropout and batch normalization for better generalization.

## Model Training
- Optimized model hyperparameters using Adam optimizer and learning rate scheduling.
- Applied **data augmentation techniques** to balance the dataset and prevent overfitting.
- Evaluated performance using accuracy, precision, recall, and F1-score.

## Web Application
- Developed an interactive **frontend using Streamlit** for real-time monitoring.
- Users can upload EEG signals and receive real-time seizure predictions.
- Integrated **PyTorch hooks** to visualize key EEG channels influencing model decisions.

## Deployment
- Used **FastAPI** to serve the trained model via a RESTful API.
- Deployed the web application on a cloud server for real-time accessibility.
- Optimized inference speed to handle real-time EEG streaming efficiently.

## Results
- Achieved **high accuracy** in detecting seizure vs. non-seizure events.
- Real-time predictions with minimal latency.
- Clear visualization of critical EEG channels contributing to each decision.
