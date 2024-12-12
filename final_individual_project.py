#!/usr/bin/env python
# coding: utf-8

# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import altair as alt

# Load data
s = pd.read_csv("social_media_usage.csv")

# Cleaning function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Apply cleaning
s['sm_li'] = clean_sm(s['web1h']) 
features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']

# Select relevant columns and rename
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
ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)]
ss = ss.dropna()

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
    X,  # Features
    y,  # Target
    test_size=0.2,  # 20% of the data is used for testing
    random_state=42,  # Ensures reproducibility
    stratify=y  # Preserves the distribution of target classes
)

# Train logistic regression model
model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("LinkedIn Usage Predictor")
st.write("This app predicts whether a person is likely to use LinkedIn based on their demographics and attributes.")

# Input fields
income_level = st.selectbox(
    "Income Level (1-9)", 
    options=[
        (1, "Less than $10,000"),
        (2, "$10,000 to $20,000"),
        (3, "$20,000 to $30,000"),
        (4, "$30,000 to $40,000"),
        (5, "$40,000 to $50,000"),
        (6, "$50,000 to $75,000"),
        (7, "$75,000 to $100,000"),
        (8, "$100,000 to $150,000"),
        (9, "$150,000 or more")
    ]
)
income = income_level[0]  # Extract numeric value

education_level = st.selectbox(
    "Education Level (1-8)", 
    options=[
        (1, "Less than high school"),
        (2, "High school incomplete"),
        (3, "High school graduate"),
        (4, "Some college, no degree"),
        (5, "Two-year associate degree"),
        (6, "Four-year college/university degree"),
        (7, "Some postgraduate or professional schooling"),
        (8, "Postgraduate or professional degree")
    ]
)
education = education_level[0]  # Extract numeric value

parent = st.radio("Are you a parent?", ("No", "Yes"))
parent = 1 if parent == "Yes" else 0
married = st.radio("Are you married?", ("No", "Yes"))
married = 1 if married == "Yes" else 0
gender = st.radio("Gender", ("Male", "Female"))
female = 1 if gender == "Female" else 0
age = st.slider("Age (18-98)", min_value=18, max_value=98, value=30)

# Create user data for prediction
user_data = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [married],
    'female': [female],
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

    # Data for chart
    chart_data = pd.DataFrame({
        'Outcome': ['LinkedIn User', 'Not LinkedIn User'],
        'Probability': pred_prob
    })

    # Custom Altair bar chart
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Outcome', sort=['LinkedIn User', 'Not LinkedIn User'], title="Outcome"),
        y=alt.Y('Probability', title="Probability"),
        color=alt.Color('Outcome', scale=alt.Scale(
            domain=['LinkedIn User', 'Not LinkedIn User'],
            range=['blue', 'grey']
        )),
        tooltip=['Outcome', 'Probability']
    ).properties(
        width=400,
        height=300,
        title="Prediction Probabilities"
    )

    st.altair_chart(bar_chart, use_container_width=True)
