
import streamlit as st
import pandas as pd
import pickle

# Load model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Missing 'model.pkl' or 'vectorizer.pkl'. Please ensure both files are in the same folder.")
    st.stop()

# Streamlit page setup
st.set_page_config(page_title="Security Match Predictor", layout="centered")
st.title("🔐 Security Match Predictor")
st.markdown("Use this app to check if two financial descriptions refer to the same security.")

# Input fields
desc_x = st.text_area("Enter Description X", height=100)
desc_y = st.text_area("Enter Description Y", height=100)

# Prediction button
if st.button("Predict"):
    if not desc_x.strip() or not desc_y.strip():
        st.warning("Please enter both descriptions.")
    else:
        combined_text = desc_x + " " + desc_y
        vectorized_input = vectorizer.transform([combined_text])
        prediction = model.predict(vectorized_input)[0]
        result = "✅ Same Security" if prediction else "❌ Different Security"
        st.subheader("Prediction Result")
        st.success(result)
