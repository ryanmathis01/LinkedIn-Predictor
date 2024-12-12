#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load model and dataset
# Assuming you already trained your logistic regression model and saved it

# Income labels
income_labels = {
    1: "Less than $10k",
    2: "$10k - $20k",
    3: "$20k - $30k",
    4: "$30k - $40k",
    5: "$40k - $50k",
    6: "$50k - $75k",
    7: "$75k - $100k",
    8: "$100k - $150k",
    9: "$150k or more"
}

# Education levels
education_labels = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year college/university degree",
    7: "Some postgraduate education",
    8: "Postgraduate or professional degree"
}

st.title("LinkedIn Usage Predictor")
st.markdown("This app predicts whether a person is likely to use LinkedIn based on their demographics and attributes.")

# Inputs
income = st.selectbox("Income Level", options=list(income_labels.keys()), format_func=lambda x: income_labels[x])
education = st.selectbox("Education Level", options=list(education_labels.keys()), format_func=lambda x: education_labels[x])
parent = st.radio("Are you a parent?", ["No", "Yes"])
married = st.radio("Are you married?", ["No", "Yes"])
gender = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age (18-98)", min_value=18, max_value=98, value=25)

# Convert inputs to binary or numerical format for model
parent_binary = 1 if parent == "Yes" else 0
married_binary = 1 if married == "Yes" else 0
gender_binary = 1 if gender == "Female" else 0

# Feature vector for prediction
user_data = [[income, education, parent_binary, married_binary, gender_binary, age]]

# Placeholder for model prediction
# Replace `model` with your actual trained model instance
model = LogisticRegression()
# Placeholder for prediction probabilities
pred_prob = [0.5, 0.5]  # Replace with actual `model.predict_proba(user_data)`

# Prediction
predicted_class = "LinkedIn User" if pred_prob[1] > 0.5 else "Not a LinkedIn User"
probability = pred_prob[1] * 100

# Display results
st.subheader("Prediction Results")
st.markdown(f"**Predicted Category:** {predicted_class}")
st.markdown(f"**Probability of being a LinkedIn User:** {probability:.1f}%")

# Display probability chart
st.bar_chart({"Outcome": ["LinkedIn User", "Not LinkedIn User"], "Probability": pred_prob})
