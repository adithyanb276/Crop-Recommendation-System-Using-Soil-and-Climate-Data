

# 🌾 Crop Recommendation System

> **Predictive Analytics — Group Project**

![Python](https://img.shields.io/badge/Python-3.10+-blue) 
![Streamlit](https://img.shields.io/badge/Streamlit-Live-green) 
![Accuracy](https://img.shields.io/badge/Accuracy-99.6%25-brightgreen)

## 👥 Team Members
| Member | Contributions |
|---|---|
| 🔴 [Your Name] | S1, S3, S5, S6, S9, S10 |
| Abhita | S2, S4, S7, S8, S10 |

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
| S1 | Problem definition & literature review | 🔴 Your Name |
| S2 | Data collection & understanding | Abhita |
| S3 | Data preprocessing & cleaning | 🔴 Your Name |
| S4 | Exploratory data analysis | Abhita |
| S5 | Feature engineering & selection | 🔴 Your Name |
| S6 | Model building & training | 🔴 Your Name |
| S7 | Model evaluation & comparison | Abhita |
| S8 | Model interpretation & explainability | Abhita |
| S9 | Streamlit deployment | 🔴 Your Name |
| S10 | Documentation & presentation | Both |

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
git clone https://github.com/🔴YOURUSERNAME/crop-recommendation-system-using-soil-and-climate-data.git
cd crop-recommendation-system-using-soil-and-climate-data
pip install -r requirements.txt
streamlit run app/app.py
```

## 📁 Repository Structure
