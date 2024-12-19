# **Preoperative VO₂max Predictions from Wearable Sensor Data**

This repository contains the code for the paper titled **"Preoperative VO₂max Predictions from Wearable Sensor Data: Exploring Short and Long-term Heart Rate Variability."** The code is divided into two main sections: **feature extraction** and **model development.**

## **Repository Overview**

### **Feature Extraction**
The feature extraction section consists of three notebooks that access multiple streams of data:

- **Step Counts**:
  - Extracted using the open-source tool: *"Open-Source, Step-Counting Algorithm for Smartphone Data Collected in Clinical and Nonclinical Settings: Algorithm Development and Validation Study".*

- **Movement Classifications**:
  - Derived using the *Oxford Biobank Tool*.

- **Heart Rate (HR) and ECG Signals**:
  - Preprocessed HR data in 10-second intervals.
  - Raw ECG signals.

#### **Notebooks in Feature Extraction**
- **Movement Metrics Notebook**:
  - Calculates movement metrics, including metrics derived from HR signals.

- **Short-Term HRV Notebook**:
  - Computes 10 short-term HRV metrics.
  - Identifies periods of sleep for HRV calculations.

- **Long-Term HRV Notebook**:
  - Computes long-term HRV metrics (e.g., SDNN24 and its variations).
  - Uses both HR data and ECG signals for analysis.

---

### **Model Development**
This section contains two main files:

- **Build File**:
  - Performs feature selection using LASSO regression.
  - Compares two Linear Regression (LR) models:
    - One including HRV metrics.
    - One excluding HRV metrics.
  - Results are evaluated across four metrics using 10-fold cross-validation.
  - Appropriate statistical tests (e.g., t-tests) are used to compare the models.

- **SHAP Notebook**:
  - Calculates SHAP (SHapley Additive exPlanations) values to determine feature contributions to the HRV model.

---

## **Data Availability**
This repository does not include any raw data. It is focused on feature extraction and model development. For raw signal processing, please refer to these links: (to be provided).



## **Disclaimer**
This repository is intended for academic and research purposes only. Ensure proper data handling practices and ethical considerations when applying this code to your own datasets.
