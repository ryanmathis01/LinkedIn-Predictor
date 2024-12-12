#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# Train and save the model
@st.cache(allow_output_mutation=True)
def train_model():
    model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# User input fields with labels
st.header("Enter Details to Predict LinkedIn Usage")

income = st.selectbox(
    "Income Level (1-9)", 
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    format_func=lambda x: {
        1: "Less than $10,000",
        2: "$10,000 to $19,999",
        3: "$20,000 to $29,999",
        4: "$30,000 to $39,999",
        5: "$40,000 to $49,999",
        6: "$50,000 to $74,999",
        7: "$75,000 to $99,999",
        8: "$100,000 to $149,999",
        9: "$150,000 or more"
    }.get(x, "Unknown")
)

education = st.selectbox(
    "Education Level (1-8)",
    options=[1, 2, 3, 4, 5, 6, 7, 8],
    format_func=lambda x: {
        1: "Less than high school",
        2: "High school incomplete",
        3: "High school graduate",
        4: "Some college, no degree",
        5: "Two-year associate degree",
        6: "Four-year college/university degree",
        7: "Some postgraduate education",
        8: "Postgraduate or professional degree"
    }.get(x, "Unknown")
)

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
    pred_prob = model.predict_proba(user_data)[0]
    predicted_class = "LinkedIn User" if pred_prob[1] > 0.5 else "Not a LinkedIn User"
    probability = pred_prob[1] * 100

    # Display results
    st.subheader("Prediction Results")
    st.markdown(f"**Predicted Category:** {predicted_class}")
    st.markdown(f"**Probability of being a LinkedIn User:** {probability:.2f}%")
    st.bar_chart(pd.DataFrame({'Outcome': ['LinkedIn User', 'Not LinkedIn User'], 
                               'Probability': pred_prob}).set_index('Outcome'))
