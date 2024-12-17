import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv(r"C:\Users\Admin\Documents\Credit_Card_Fraud_Detection\creditcard.xlsx")  # Ensure this is the correct file format

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input('Enter all required feature values, separated by commas')

submit = st.button("Submit")

if submit:
    try:
        # Split and convert input into float
        input_df_splited = input_df.split(',')
        features = np.asarray(input_df_splited, dtype=np.float64)
        
        # Ensure correct shape
        prediction = model.predict(features.reshape(1, -1))
        
        if prediction[0] == 0:
            st.write("Legitimate Transaction")
        else:
            st.write("Fraudulent Transaction")
    except ValueError:
        st.write("Please enter valid numerical values for all features.")
    except Exception as e:
        st.write(f"Error: {e}")
