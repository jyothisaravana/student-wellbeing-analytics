import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --- Load trained models ---
with open('models/mood_regressor.pkl', 'rb') as f:
    mood_model = pickle.load(f)
with open('models/academic_regressor.pkl', 'rb') as f:
    academic_model = pickle.load(f)
with open('models/stress_classifier.pkl', 'rb') as f:
    stress_model = pickle.load(f)

st.set_page_config(page_title="Student Wellbeing App", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Single Prediction", "Analysis Dashboard", "About"])

# --- Single Prediction Form ---
if page == "Single Prediction":
    st.title("Predict Student Mood, Academic Score & Stress Risk")
    sleep = st.slider("Sleep (hrs)", 4, 10, 7)
    study = st.slider("Study Hours", 0, 8, 3)
    stress = st.slider("Stress (1-10)", 1, 10, 5)
    exercise = st.slider("Exercise (mins)", 0, 60, 15)
    screen = st.slider("Screen Time (hrs)", 0, 10, 4)
    social = st.slider("Social Score (1-5)", 1, 5, 3)
    mood = st.slider("Mood (1-10)", 1, 10, 5)

    # Prepare DataFrames for prediction with correct column names
    features_reg = ['Sleep (hrs)', 'Study Hours', 'Stress', 'Exercise (mins)', 'Screen Time (hrs)', 'Social Score']
    user_input_reg = pd.DataFrame([[sleep, study, stress, exercise, screen, social]], columns=features_reg)

    features_clf = ['Sleep (hrs)', 'Study Hours', 'Exercise (mins)', 'Screen Time (hrs)', 'Social Score', 'Mood']
    user_input_clf = pd.DataFrame([[sleep, study, exercise, screen, social, mood]], columns=features_clf)

    if st.button("Predict Mood & Academic Score"):
        predicted_mood = mood_model.predict(user_input_reg)[0]
        predicted_score = academic_model.predict(user_input_reg)[0]
        st.success(f"Predicted Mood: {predicted_mood:.2f}")
        st.success(f"Predicted Academic Score: {predicted_score:.2f}")
    
    if st.button("Predict High Stress Day"):
        high_stress = stress_model.predict(user_input_clf)[0]
        prob = stress_model.predict_proba(user_input_clf)[0][1]
        if high_stress:
            st.warning(f"High Stress Day Risk! (Confidence: {prob:.2%})")
        else:
            st.info(f"Low Stress Day Risk. (Confidence: {prob:.2%})")

# --- Analysis Dashboard ---
elif page == "Analysis Dashboard":
    st.title("Analysis Dashboard")
    importance = pd.Series(
        mood_model.coef_, 
        index=['Sleep (hrs)', 'Study Hours', 'Stress', 'Exercise (mins)', 'Screen Time (hrs)', 'Social Score']
    )
    st.subheader("Mood Model Feature Importances")
    fig, ax = plt.subplots()
    importance.sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)

    # Add more visuals, stats, or interpretation as you like!

# --- About Page ---
elif page == "About":
    st.title("About This App")
    st.write("""
    This application predicts student mood, academic score, and stress risk using real-world data and machine learning models.
    Built with Python, pandas, scikit-learn, and Streamlit.
    """)
