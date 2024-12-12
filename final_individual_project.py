#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# In[17]:


# Load data
s = pd.read_csv("social_media_usage.csv")
s.shape


# In[19]:


# Cleaning function
def clean_sm(x):
    return np.where(x == 1, 1, 0)


# In[23]:


# Q2: Test the clean_sm function with a toy dataframe
toy_df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'status': [1, 0, 2]
})
s.columns


# In[25]:


# Create a new dataframe with cleaned data
s['sm_li'] = clean_sm(s['web1h']) 
features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']


# In[27]:


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


# In[29]:


# Remove outliers and missing values
ss = ss[(ss['income'] <= 9) & (ss['education'] <= 8) & (ss['age'] <= 98)]
ss = ss.dropna()


# In[31]:


# Binary transformations
ss['female'] = np.where(ss['gender'] == 2, 1, 0)  # Female = 1, Male = 0
ss['married'] = np.where(ss['married'] == 1, 1, 0)  # Married = 1, Otherwise = 0
ss = ss.drop(columns=['gender'])  # Drop 'gender' column after transformation


# In[33]:


# Prepare features and target
features = ['income', 'education', 'parent', 'married', 'female', 'age']
X = ss[features]
y = ss['sm_li']


# In[45]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Features
    y,  # Target
    test_size=0.2,  # 20% of the data is used for testing
    random_state=42,  # Ensures reproducibility
    stratify=y  # Preserves the distribution of target classes
    )
# Explanation of each object:
# 1. X_train: Features used for training the model. Contains 80% of the total data.
#    Purpose: Used by the model to learn the relationship between features and the target variable.
# 2. X_test: Features used for evaluating the model. Contains 20% of the total data.
#    Purpose: Input for the model during testing to evaluate its predictive accuracy.
# 3. y_train: Target labels corresponding to X_train. Contains 80% of the total labels.
#    Purpose: Used during training to teach the model the correct labels for the training features.
# 4. y_test: Target labels corresponding to X_test. Contains 20% of the total labels.
#    Purpose: Used to assess the model's performance by comparing predictions to true labels.

# Print the sizes of the splits for confirmation
print(f"Training set size: {X_train.shape}, Training labels: {y_train.shape}")
print(f"Testing set size: {X_test.shape}, Testing labels: {y_test.shape}")


# In[37]:


# Train logistic regression model
model = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
model.fit(X_train, y_train)


# In[39]:


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)


# In[41]:


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     columns=["Predicted Negative", "Predicted Positive"], 
                     index=["Actual Negative", "Actual Positive"])


# In[43]:


# Output results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm_df)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[47]:


# Hypothetical Predictions
person1 = [[8, 7, 0, 1, 1, 42]]  # Income=8, Education=7, Non-parent, Married Female, Age=42
person2 = [[8, 7, 0, 1, 1, 82]]  # Same as above but Age=82
prob1 = model.predict_proba(person1)[0][1]
prob2 = model.predict_proba(person2)[0][1]
print(f"Person 1 (42 years old): Probability of LinkedIn use: {prob1:.2f}")
print(f"Person 2 (82 years old): Probability of LinkedIn use: {prob2:.2f}")


# In[ ]:




