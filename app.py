import streamlit as st
import joblib

# Load models and vectorizer
nb_model = joblib.load("nb_model.pkl")
log_model = joblib.load("log_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title
st.title("IMDB Review Sentiment Analyzer")
st.write("Compare predictions from Naive Bayes and Logistic Regression models.")

# Input review
user_input = st.text_area("Enter your movie review:")

if user_input:
    vec_input = vectorizer.transform([user_input])
    
    nb_result = nb_model.predict(vec_input)[0]
    log_result = log_model.predict(vec_input)[0]
    
    sentiment_map = {1: "Positive", 0: "Negative"}
    
    st.subheader("Model Predictions:")
    st.write(f"**Naive Bayes:** {sentiment_map[nb_result]}")
    st.write(f"**Logistic Regression:** {sentiment_map[log_result]}")
