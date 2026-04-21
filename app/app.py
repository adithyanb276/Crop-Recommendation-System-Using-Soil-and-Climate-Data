import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ── Load models ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf    = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le    = joblib.load('models/label_encoder.pkl')
    return rf, scaler, le

rf_model, scaler, le = load_models()
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# ── Header ───────────────────────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown("Enter your soil and climate data below to get an instant crop recommendation powered by Machine Learning.")
st.divider()

# ── Sidebar inputs ───────────────────────────────────────────────────────
st.sidebar.header("🧪 Enter Soil & Climate Data")

N   = st.sidebar.slider("Nitrogen (N) — kg/ha",       0,   140, 50)
P   = st.sidebar.slider("Phosphorus (P) — kg/ha",     5,   145, 50)
K   = st.sidebar.slider("Potassium (K) — kg/ha",      5,   205, 48)
temp = st.sidebar.slider("Temperature — °C",           8.0, 44.0, 25.0, step=0.1)
hum  = st.sidebar.slider("Humidity — %",              14.0, 100.0, 71.0, step=0.1)
ph   = st.sidebar.slider("pH Value",                   3.5,  9.9,  6.5, step=0.1)
rain = st.sidebar.slider("Rainfall — mm",             20.0, 299.0, 103.0, step=0.1)

predict_btn = st.sidebar.button("🔍 Recommend Crop", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([[N, P, K, temp, hum, ph, rain]], columns=features)

    # Predict
    proba = rf_model.predict_proba(input_data)[0]
    pred_idx = np.argmax(proba)
    pred_crop = le.inverse_transform([pred_idx])[0]
    confidence = proba[pred_idx] * 100

    # ── Result banner ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(f"### ✅ Recommended Crop")
        st.markdown(f"## 🌱 **{pred_crop.upper()}**")
        st.metric("Confidence", f"{confidence:.1f}%")

    with col2:
        st.markdown("#### Your Input Summary")
        input_display = pd.DataFrame({
            'Feature': ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall'],
            'Value': [N, P, K, temp, hum, ph, rain],
            'Unit': ['kg/ha', 'kg/ha', 'kg/ha', '°C', '%', '', 'mm']
        })
        st.dataframe(input_display, hide_index=True, use_container_width=True)

    st.divider()

    # ── Probability chart ────────────────────────────────────────────────
    st.markdown("#### Prediction Probabilities — All 22 Crops")
    prob_df = pd.DataFrame({
        'Crop': le.classes_,
        'Probability': proba * 100
    }).sort_values('Probability', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2ecc71' if c == pred_crop else '#bdc3c7' for c in prob_df['Crop']]
    ax.barh(prob_df['Crop'], prob_df['Probability'], color=colors, edgecolor='white')
    ax.set_xlabel('Probability (%)')
    ax.set_title('Crop Recommendation Probabilities', fontsize=13)
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for i, (crop, prob) in enumerate(zip(prob_df['Crop'], prob_df['Probability'])):
        if prob > 1:
            ax.text(prob + 0.3, i, f'{prob:.1f}%', va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── SHAP explanation ─────────────────────────────────────────────────
    st.markdown("#### 🔍 Why this crop? — SHAP Explanation")
    st.caption("SHAP values show how each feature pushed the prediction toward or away from the recommended crop.")

    try:
        explainer = shap.TreeExplainer(rf_model)
        shap_exp = explainer(input_data)

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        shap.waterfall_plot(shap_exp[0, :, pred_idx], max_display=7, show=False)
        plt.title(f'SHAP Waterfall — Why {pred_crop}?', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

else:
    st.info("👈 Adjust the sliders on the left and click **Recommend Crop** to get started.")

    # Show sample data
    st.markdown("#### 📋 Sample crop requirements (from dataset)")
    sample_crops = {
        'Crop': ['rice', 'maize', 'chickpea', 'mango', 'coffee'],
        'N': [90, 77, 40, 20, 101],
        'P': [42, 48, 67, 27, 28],
        'K': [43, 48, 79, 30, 29],
        'Temperature': [21, 22, 18, 31, 25],
        'Humidity': [82, 65, 16, 50, 58],
        'pH': [6.5, 6.0, 7.3, 5.8, 6.5],
        'Rainfall': [203, 82, 83, 94, 158]
    }
    st.dataframe(pd.DataFrame(sample_crops), hide_index=True, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.caption("Predictive Analytics Group Project · Random Forest · KNN · Decision Tree · Built with Streamlit")
