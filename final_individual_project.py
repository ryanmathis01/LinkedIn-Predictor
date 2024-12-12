#!/usr/bin/env python
# coding: utf-8
# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Title for the Streamlit App
st.title("LinkedIn Usage Predictor")

# Description
st.write("This app predicts whether a person is likely to use LinkedIn based on their demographics and attributes.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("social_media_usage.csv")

s = load_data()

# Cleaning function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Process the dataset
s['sm_li'] = clean_sm(s['web1h'])
features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']

ss = s[features + ['sm_li']].copy()
rename_dict = {
    'income': 'income',
    'educ2': 'education',
    'par': 'parent',
    'marital': 'married',
    'gender': 'gender',
    'age': 'age'
}
ss.rename(columns=rename_dict, inplace=True)

# Remove outliers and missing values
ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)].dropna()

# Binary transformations
ss['female'] = np.where(ss['gender'] == 2, 1, 0)  # Female = 1, Male = 0
ss['married'] = np.where(ss['married'] == 1, 1, 0)  # Married = 1, Otherwise = 0
ss = ss.drop(columns=['gender'])  # Drop 'gender' column after transformation

# Prepare features and target
features = ['income', 'education', 'parent', 'married', 'female', 'age']
X = ss[features]
y = ss['sm_li']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
model.fit(X_train, y_train)

# Add interactive inputs for user data
st.header("Enter Details to Predict LinkedIn Usage")
income = st.number_input("Income (1-9)", min_value=1, max_value=9, step=1)
education = st.number_input("Education Level (1-8)", min_value=1, max_value=8, step=1)
parent = st.radio("Are you a parent?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
married = st.radio("Are you married?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
female = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
age = st.number_input("Age (1-98)", min_value=1, max_value=98, step=1)

# Prediction
if st.button("Predict"):
    user_input = np.array([[income, education, parent, married, female, age]])
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]

    st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of using LinkedIn: {prob:.2f}")

# Display model evaluation
if st.checkbox("Show Model Evaluation"):
    st.subheader("Model Evaluation Metrics")
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=["Predicted Negative", "Predicted Positive"], 
                         index=["Actual Negative", "Actual Positive"])
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.dataframe(cm_df)
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))




