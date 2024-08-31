import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit as st

# Load and prepare your dataset
@st.cache
def load_data():
    df = pd.read_csv('spam_emails.csv')
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# Build the pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear'))
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Add custom CSS for background color
st.markdown("""
    <style>
    .reportview-container {
        background-color: #ADD8E6; /* Light gray background */
    }
    .sidebar .sidebar-content {
        background-color: #d1e0e0; /* Light teal background for sidebar */
    }
    h1 {
        color: red; /* Red color for headings */
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
st.markdown("<h1>Spam Email Classifier</h1>", unsafe_allow_html=True)
st.write("Enter the email text below to classify it as spam or not spam:")

# Input text
input_text = st.text_area("Email Text", "")

# Prediction
if st.button('Classify'):
    if input_text:
        prediction = pipeline.predict([input_text])
        result = prediction[0]
        st.write(f"The email is classified as: **{result}**")
    else:
        st.write("Please enter some text.")
