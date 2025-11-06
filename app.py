import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("student_model.pkl", "rb"))

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Prediction Web App")

st.write("Predict student performance based on demographics, study habits, and parental background.")

# Input fields
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_education = st.selectbox("Parental Education Level", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading = st.number_input("Reading Score (0-100)", min_value=0, max_value=100, value=70)
writing = st.number_input("Writing Score (0-100)", min_value=0, max_value=100, value=70)

# Encode input manually (same order as training)
input_data = pd.DataFrame({
    'gender': [0 if gender == "female" else 1],
    'race/ethnicity': [ord(race[-1]) - ord('A')],
    'parental level of education': [
        ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"].index(parent_education)
    ],
    'lunch': [0 if lunch == "free/reduced" else 1],
    'test preparation course': [0 if prep_course == "none" else 1],
    'reading score': [reading],
    'writing score': [writing]
})

if st.button("Predict Performance"):
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Math Score: {prediction[0]:.2f}")

