# Real-Time Seizure Detection Web Application
[Link to the Web App](https://eeg-detection-vr8h.onrender.com)
[Demo Video](https://drive.google.com/file/d/1Gqmdc3Q_KLzb-DW38f0wK6QzUNpMK-JP/view?usp=sharing)
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
Our solution leverages deep learning models to analyze EEG data and detect seizures in real time. The key steps involved are:
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
- Developed an interactive **frontend using HTML CSS and JavaScript** for real-time monitoring.
- Users can upload EEG signals and receive real-time seizure predictions.
- Integrated **PyTorch hooks** to visualize key EEG channels influencing model decisions.

## Deployment
- Used **FastAPI** to serve the trained model via a RESTful API.
- Deployed the web application on a cloud server for real-time accessibility.
- Optimized inference speed to handle real-time EEG streaming efficiently.

## Results
The model achieved outstanding performance in detecting seizure vs. non-seizure events, as demonstrated by the classification report:

### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Normal) | 0.98 | 0.99 | 0.98 | 696 |
| 1 (Complex Partial) | 0.99 | 0.97 | 0.98 | 549 |
| 2 (Electrographic) | 0.99 | 0.99 | 0.99 | 137 |
| 3 (Video-detected) | 1.00 | 1.00 | 1.00 | 21 |

- **Overall Accuracy:** 98% (1403 samples)
- **Macro Average:** Precision: 0.99 | Recall: 0.99 | F1-Score: 0.99 | Support: 1403 samples
- **Weighted Average:** Precision: 0.98 | Recall: 0.98 | F1-Score: 0.98 | Support: 1403 samples
- **Balanced Accuracy:** 98.76%
- **AUC-ROC Score:** 0.9983

The model demonstrates high accuracy and generalization capability, with a near-perfect ability to distinguish between seizure and non-seizure events.

