# ui/app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout="wide")

# --- 1. Load Model and Define Columns ---
try:
    pipeline = joblib.load('../models/final_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Make sure 'final_model.pkl' is in the 'models' directory.")
    st.stop()

# This is YOUR exact list of columns. The app will now work correctly.
MODEL_COLUMNS = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_1.0', 'cp_2.0', 'cp_3.0', 
    'cp_4.0', 'fbs_1.0', 'restecg_1.0', 'restecg_2.0', 'exang_1.0', 'slope_2.0', 
    'slope_3.0', 'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_6.0', 'thal_7.0'
]

# --- 2. App Title and Description ---
st.title('❤️ Heart Disease Prediction App')
st.write("This app predicts the likelihood of a patient having heart disease based on their medical data.")
st.write("---")

# --- 3. User Input Collection in Sidebar ---
st.sidebar.header('Patient Medical Data')

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 54)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 132)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 246)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)

    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ('False', 'True'))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', ('Normal', 'ST-T Wave Abnormality', 'Probable or Definite Left Ventricular Hypertrophy'))
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ('Upsloping', 'Flat', 'Downsloping'))
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Thalassemia', ('Normal', 'Fixed Defect', 'Reversible Defect'))

    # --- 4. Preprocess User Input for Model ---
    model_input = {col: 0 for col in MODEL_COLUMNS}

    model_input['age'] = age
    model_input['trestbps'] = trestbps
    model_input['chol'] = chol
    model_input['thalach'] = thalach
    model_input['oldpeak'] = oldpeak

    if sex == 'Male': model_input['sex_1.0'] = 1
    if cp == 'Atypical Angina': model_input['cp_2.0'] = 1
    if cp == 'Non-anginal Pain': model_input['cp_3.0'] = 1
    if cp == 'Asymptomatic': model_input['cp_4.0'] = 1
    if fbs == 'True': model_input['fbs_1.0'] = 1
    if restecg == 'ST-T Wave Abnormality': model_input['restecg_1.0'] = 1
    if restecg == 'Probable or Definite Left Ventricular Hypertrophy': model_input['restecg_2.0'] = 1
    if exang == 'Yes': model_input['exang_1.0'] = 1
    if slope == 'Flat': model_input['slope_2.0'] = 1
    if slope == 'Downsloping': model_input['slope_3.0'] = 1
    if ca == 1: model_input['ca_1.0'] = 1
    if ca == 2: model_input['ca_2.0'] = 1
    if ca == 3: model_input['ca_3.0'] = 1
    
    # Correctly map Thalassemia. 'Normal' is the base case (both are 0).
    if thal == 'Fixed Defect': model_input['thal_6.0'] = 1
    if thal == 'Reversible Defect': model_input['thal_7.0'] = 1

    input_df = pd.DataFrame([model_input])[MODEL_COLUMNS]
    return input_df

input_df = user_input_features()

st.subheader('Processed Input Data Sent to Model')
st.write(input_df)

# --- 5. Prediction and Display Results ---
if st.sidebar.button('Predict'):
    try:
        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)

        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.error('**High Risk** of Heart Disease Detected')
        else:
            st.success('**Low Risk** of Heart Disease Detected')

        st.subheader('Prediction Probability')
        st.write(f"The model predicts a **{prediction_proba[0][1]*100:.2f}%** probability of the patient having heart disease.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")