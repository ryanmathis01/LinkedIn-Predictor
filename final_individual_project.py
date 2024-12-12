#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Title of the app
st.title("LinkedIn Usage Predictor")
st.markdown("This app predicts whether a person is likely to use LinkedIn based on their demographics and attributes.")

# Load data
@st.cache
def load_data():
    s = pd.read_csv("social_media_usage.csv")

    # Cleaning and feature engineering
    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    s['sm_li'] = clean_sm(s['web1h'])
    features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']
    ss = s[features + ['sm_li']].copy()

    # Rename columns
    ss.rename(columns={
        'income': 'income',
        'educ2': 'education',
        'par': 'parent',
        'marital': 'married',
        'gender': 'gender',
        'age': 'age'
    }, inplace=True)

    # Remove outliers and missing values
    ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)]
    ss = ss.dropna()

    # Transform binary columns
    ss['female'] = np.where(ss['gender'] == 2, 1, 0)  # Female = 1, Male = 0
    ss['married'] = np.where(ss['married'] == 1, 1, 0)  # Married = 1, Otherwise = 0
    ss.drop(columns=['gender'], inplace=True)

    return ss

# Load and prepare data
data = load_data()
features = ['income', 'education', 'parent', 'married', 'female', 'age']
X = data[features]
y = data['sm_li']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and save the model if not already saved
@st.cache(allow_output_mutation=True)
def train_and_save_model():
    model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_and_save_model()

# User input fields
st.header("Enter Details to Predict LinkedIn Usage")

income = st.number_input("Income (1-9)", min_value=1, max_value=9, value=5)
education = st.number_input("Education Level (1-8)", min_value=1, max_value=8, value=4)
parent = st.radio("Are you a parent?", ["No", "Yes"])
married = st.radio("Are you married?", ["No", "Yes"])
gender = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age (1-98)", min_value=1, max_value=98, value=25)

# Convert user inputs into a dataframe
parent_val = 1 if parent == "Yes" else 0
married_val = 1 if married == "Yes" else 0
female_val = 1 if gender == "Female" else 0

user_data = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent_val],
    'married': [married_val],
    'female': [female_val],
    'age': [age]
})

# Prediction button
if st.button("Predict LinkedIn Usage"):
    try:
        pred_prob = model.predict_proba(user_data)[0]
        predicted_class = "LinkedIn User" if pred_prob[1] > 0.5 else "Not a LinkedIn User"
        probability = pred_prob[1] * 100

        # Display results
        st.subheader("Prediction Results")
        st.markdown(f"**Predicted Category:** {predicted_class}")
        st.markdown(f"**Probability of being a LinkedIn User:** {probability:.2f}%")
        st.bar_chart(pd.DataFrame({'Outcome': ['LinkedIn User', 'Not LinkedIn User'], 
                                   'Probability': pred_prob}).set_index('Outcome'))
    except Exception as e:
        st.error(f"An error occurred: {e}")
