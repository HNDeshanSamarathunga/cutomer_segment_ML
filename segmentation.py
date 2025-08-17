import streamlit as st
import pandas as pd
import numpy as np
import joblib


kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Segmentation Analysis")
st.write("Enter customer details to predict the segment.")

age =  st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
total_spending = 
num_web_purchases = 
num_store_purchases = 
num_web_visits = 
recency = 