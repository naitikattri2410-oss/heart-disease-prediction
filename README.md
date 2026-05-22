# 🫀 Heart Disease Prediction

A machine learning project that predicts the presence of heart disease using clinical patient data. Built with Python and scikit-learn, achieving **87.80% accuracy** using Logistic Regression.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction using patient clinical data can significantly improve outcomes. This project builds a binary classifier to predict whether a patient has heart disease based on 13 clinical features.

---

## 📊 Dataset

- **Source:** UCI Heart Disease Dataset (combined Cleveland, Hungarian, Switzerland, Long Beach VA)
- **Rows:** 1,024 patient records
- **Target:** `target_binary` — 0 = No Heart Disease, 1 = Heart Disease
- **Class Distribution:** 554 (No Disease) / 470 (Disease)

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient (years) |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia type |

---

## 🤖 Model & Results

| Metric | Score |
|---|---|
| **Accuracy** | **87.80%** |
| Precision (No Disease) | 88% |
| Precision (Disease) | 88% |
| Recall (No Disease) | 90% |
| Recall (Disease) | 85% |
| F1-Score (macro avg) | 0.88 |

**Algorithm:** Logistic Regression with StandardScaler preprocessing  
**Train/Test Split:** 80% / 20% (random_state=42)

---

## 📁 Project Structure

```
heart-disease-prediction/
├── heart_disease.csv       # Dataset
├── heart.py                # Main ML script
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python heart.py
```

This will:
- Print dataset info and null value counts
- Display a class distribution bar chart
- Show a feature correlation heatmap
- Train the Logistic Regression model
- Print model accuracy
- Display the confusion matrix

---

## 📈 Visualizations

The script generates:
- **Count Plot** — distribution of heart disease vs. no heart disease
- **Correlation Heatmap** — relationships between all 13 features
- **Confusion Matrix** — breakdown of true/false positives and negatives

---

## 🛠 Tech Stack

- **Python 3.8+**
- **pandas** — data loading and manipulation
- **scikit-learn** — model training, evaluation, preprocessing
- **matplotlib / seaborn** — data visualization

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
