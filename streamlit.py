import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("pipeline_diabetes.pkl")

st.title("📈Prediksi Risiko Diabetes📈")

# Input dari pengguna
pregnancies = st.number_input("Pregnancies", 0)
glucose = st.number_input("Glucose", 0.0)
blood_pressure = st.number_input("Blood Pressure", 0.0)
skin_thickness = st.number_input("Skin Thickness", 0.0)
insulin = st.number_input("Insulin", 0.0)
bmi = st.number_input("BMI", 0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0)
age = st.number_input("Age", 0)

# Buat turunan fitur
if insulin <= 80:
    insulin_level = "Low"
elif insulin <= 120:
    insulin_level = "Normal"
else:
    insulin_level = "Prediabet"

if bmi < 18.5:
    weight_category = "Underweight"
elif bmi < 25:
    weight_category = "Normal"
elif bmi < 30:
    weight_category = "Overweight"
else:
    weight_category = "Obese"

# Susun dataframe input
input_df = pd.DataFrame([{
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
    "WeightCategory": weight_category,
    "InsulinLevel": insulin_level
}])

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error(f"Hasil Prediksi: Positif Diabetes (Probabilitas: {prob:.2f}%)")
    else:
        st.success(f"Hasil Prediksi: Negatif Diabetes (Probabilitas: {prob:.2f}%)")
