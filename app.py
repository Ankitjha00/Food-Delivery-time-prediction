import os
import sys
from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Utils.main_utils import load_object
from FoodDeliveryTimePrediction.Utils.main_utils import TimePredictionModel
import streamlit as st
import pandas as pd



model_path = os.path.join('final_model', 'model.pkl')
model = load_object(model_path)
preprocessor_path=os.path.join('final_model','preprocessor.pkl')
preprocessor = load_object(preprocessor_path)
final_model=TimePredictionModel(preprocessor=preprocessor,model=model)
    

# Streamlit UI
st.set_page_config(page_title="Food Delivery Time Prediction", layout="centered")

st.title("🍔 Food Delivery Time Prediction 🚴‍♂️")
st.write("Enter the order details below to predict the delivery time.")

# Input form
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (in km)", min_value=0.1, max_value=50.0, value=5.0)
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Snowy", "Foggy", "Windy"])
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

with col2:
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter", "Car"])
    preparation_time = st.number_input("Preparation Time (min)", min_value=1, max_value=60, value=15)
    # courier_experience = st.number_input("Courier Experience (yrs)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

if st.button("Predict Delivery Time"):
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([[
            distance,
            weather,
            traffic,
            time_of_day,
            vehicle,
            preparation_time
        ]], columns=[
            "Distance_km",
            "Weather",
            "Traffic_Level",
            "Time_of_Day",
            "Vehicle_Type",
            "Preparation_Time_min",
        ])

        # Predict
        prediction = final_model.predict(input_data)
        st.success(f"⏱️ Estimated Delivery Time: **{prediction} minutes**")

    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)