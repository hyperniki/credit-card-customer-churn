# Deploy model
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("./model/churn_model.joblib")

model = load_model()

st.title("Customer churn demo")
uploaded = st.file_uploader("Upload a CSV file with customer data", type="csv")

if uploaded is not None:
    customers = pd.read_csv(uploaded)
    probabilities = model.predict_proba(customers)[:, 1]
    results = customers.assign(churn_probability=probabilities)
    st.write("Predicted churn probabilities for uploaded customers:")
    st.dataframe(results)