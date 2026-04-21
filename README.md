

# 🌾 Crop Recommendation System

> **Predictive Analytics — Group Project**

![Python](https://img.shields.io/badge/Python-3.10+-blue) 
![Streamlit](https://img.shields.io/badge/Streamlit-Live-green) 
![Accuracy](https://img.shields.io/badge/Accuracy-99.6%25-brightgreen)

## 👥 Team Members
| Member | Contributions |
|---|---|
|  Adithyan Biju  | S1, S3, S5, S6, S9, S10 |
| Abhitha Raj | S2, S4, S7, S8, S10 |

## 🌐 Live App
### 👉 [Click here to open the Streamlit App ]   (https://share.streamlit.io/app/crop-recommendation-system-bjr5ew6ggqbgvvynj2vvdo/)

---

## 📌 Problem Statement
Farmers often lack data-driven guidance on which crop to grow based on 
their soil and climate conditions. This project builds an AI-powered 
crop recommendation system that predicts the most suitable crop using 
7 measurable parameters — Nitrogen, Phosphorus, Potassium, Temperature, 
Humidity, pH, and Rainfall.

## 📊 Dataset
| Property | Details |
|---|---|
| Source | Kaggle — Crop Recommendation Dataset |
| Rows | 2200 |
| Columns | 8 (7 features + 1 target) |
| Classes | 22 crops |
| Balance | Perfectly balanced — 100 samples per crop |
| Missing values | None |
| Duplicates | None |

## 🔬 Methodology — Data Science Life Cycle

| Stage | Description | Lead |
|---|---|---|
| S1 | Problem definition & literature review |  Adithyan Biju |
| S2 | Data collection & understanding | Abhitha Raj |
| S3 | Data preprocessing & cleaning |  Adithyan Biju |
| S4 | Exploratory data analysis | Abhitha Raj |
| S5 | Feature engineering & selection |  Adithyan Biju |
| S6 | Model building & training |  Adithyan Biju |
| S7 | Model evaluation & comparison |  Abhitha Raj |
| S8 | Model interpretation & explainability |  Abhitha Raj |
| S9 | Streamlit deployment | Adithyan Biju |
| S10 | Documentation & presentation | Adithyan Biju , Abhitha Raj |

## 📈 Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| 🥇 Random Forest | 99.60% | 🔴 (fill after running S7) |
| 🥈 Decision Tree | 98.58% | 🔴 (fill after running S7) |
| 🥉 KNN | 97.90% | 🔴 (fill after running S7) |

**Best model: Random Forest** with 99.60% cross-validation accuracy

### Key Findings
- 🌧️ **Rainfall** and **Humidity** are the most important features
- 🌿 **Potassium (K)** ranks 3rd in importance
- ⚗️ **pH** has the least influence on crop prediction
- ✅ Dataset is perfectly balanced — no class imbalance handling needed

## 🖼️ App Screenshots

### Welcome Screen
![Welcome Screen](screenshots/welcome.png)

### Prediction — Rice recommended with 90% confidence
![Prediction](screenshots/prediction.png)

### SHAP Explanation
![SHAP](screenshots/shap.png)

## ⚙️ Run Locally
```bash
git clone https://github.com/adithyanb276/crop-recommendation-system-using-soil-and-climate-data.git
cd crop-recommendation-system-using-soil-and-climate-data
pip install -r requirements.txt
streamlit run app/app.py
```

## 📁 Repository Structure


crop-recommendation-system/
├── crop_recommendation.ipynb   ← Main notebook (all 10 stages)
├── app/
│   └── app.py                  ← Streamlit app
├── data/
│   └── Crop_recommendation.csv
├── models/                     ← Saved model files
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── individual_profiles/        ← GitHub activity screenshots
├── requirements.txt
└── README.md

## 🔗 Links
- 📱 Live App: https://share.streamlit.io/app/crop-recommendation-system-bjr5ew6ggqbgvvynj2vvdo/
- 📓 Notebook: crop_recommendation.ipynb
- 📊 Dataset: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

---
*Predictive Analytics Group Project · Random Forest · KNN · 
Decision Tree · Streamlit · SHAP*


