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
num_web_visits = st.number_input("Number of Web Visits", min_value=0, max_value=100, value=100)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# Use the exact training feature names and order
feature_order = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
]

input_data = pd.DataFrame(
    [[age, income, total_spending, num_web_purchases, num_store_purchases, num_web_visits, recency]],
    columns=feature_order
)

# Cluster labels to show
cluster_labels = {
    0: "High Income, High Spending → Premium Customer",
    2: "High Web Purchases, Low Store Purchases → Digital Customer",
    5: "Low recency / inactive → Dormant Customer",
    6: "Low Income, Low Spending → Budget Customers",
}

input_data_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = int(kmeans.predict(input_data_scaled)[0])
    st.success(f"Predicted Segment: Cluster {cluster}")
    st.write(cluster_labels.get(cluster, "No label defined for this cluster."))
