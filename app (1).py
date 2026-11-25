import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Risiko Diabetes", page_icon="🩺")

@st.cache_resource
def load_model():
    return joblib.load("model_diabetes.joblib")

model = load_model()

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.title("🩺 Prediksi Risiko Diabetes")
st.write("Masukkan data pasien untuk memprediksi risiko diabetes. "
         "Hanya untuk keperluan tugas kuliah, bukan diagnosis medis.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, value=1, step=1)
    glucose = st.number_input("Glukosa (mg/dL)", min_value=0.0, value=120.0)
    blood_pressure = st.number_input("Tekanan Darah (mm Hg)", min_value=0.0, value=70.0)
    skin = st.number_input("Ketebalan Kulit (mm)", min_value=0.0, value=20.0)
with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
    age = st.number_input("Usia (tahun)", min_value=1, value=30, step=1)

if st.button("Prediksi", type="primary"):
    data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin,
                          insulin, bmi, pedigree, age]], columns=FEATURES)
    data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, np.nan)
    prob = float(model.predict_proba(data)[0][1])

    if prob >= 0.5:
        st.error(f"Berisiko Diabetes (probabilitas {prob:.1%})")
    else:
        st.success(f"Risiko Rendah (probabilitas diabetes {prob:.1%})")
    st.progress(prob)
