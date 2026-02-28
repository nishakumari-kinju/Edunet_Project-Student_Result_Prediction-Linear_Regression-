import streamlit as st
import numpy as np 
import joblib 
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl") 

st.title("Student Exam Score Predictor ") 
 
study_hours = st.slider("Total Study Hours Per Day", 0.0, 10.0, 2.0) 
self_study_hours = st.slider("Self Study Hours Per Day", 0.0, 4.0, 1.0)
social_media_hours = st.slider("Social Media Usage Per Day", 0.0, 6.0, 2.0)
gaming_hours = st.slider("Gaming Hours Per Day", 0.0, 8.0, 1.0)
sleep_hours = st.slider("Sleep Hours Per Day", 0.0, 10.0, 6.0)
part_time_job = st.selectbox ("Part-Time Job", ["No","Yes"] )
screen_time_hours = st.slider("Total Screen Hours Per Day", 0.0, 10.0, 2.0)
mental_health_score = st.slider("Mental_Health_Score", 0.0, 100.0, 25.0) 

ptj_encoded = 1 if part_time_job == "Yes" else 0 

if st.button("Predict Exam Score"): 

  input_data = np.array({[study_hours, self_study_hours, social_media_hours,gaming_hours,sleep_hours,ptj_encoded,screen_time_hours,mental_health_score]})
  prediction = model.predict(input_data)[0]

prediction = max(0, min(100,prediction)) 

st.success (f"Predicted Exam Score: {prediction:.2f}")