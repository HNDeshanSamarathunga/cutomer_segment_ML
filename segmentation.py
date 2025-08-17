import streamlit as st
import pandas as pd
import numpy as np
import joblib


kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Segmentation Analysis")
st.write("Enter customer details to predict the segment.")

age =  st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=0, max_value=1000000, value=50000)
total_spending = st.number_input("Total Spending(Sum of Purchases)", min_value=0, max_value=5000, value=1000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=1000, value=5)
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=50, value=100)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'total_spending': [total_spending],
    'num_web_purchases': [num_web_purchases],
    'num_store_purchases': [num_store_purchases],
    'num_web_visits': [num_web_visits],
    'recency': [recency]
})

input_data_scaled = scaler.transform(input_data)
