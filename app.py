from sklearn.metrics.pairwise import cosine_similarity
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
st.set_page_config(page_title="AI-Based Plagiarism Detector", layout="centered")
st.title("🧠 AI-Based Plagiarism Detector")
st.markdown("Use this app to detect potential plagiarism between two text inputs using machine learning.")


# Input fields
desc_x = st.text_area("Enter Description X", height=100)
desc_y = st.text_area("Enter Description Y", height=100)

# Prediction button
if st.button("Predict"):
    if not desc_x.strip() or not desc_y.strip():
        st.warning("Please enter both descriptions.")
    else:
       from sklearn.metrics.pairwise import cosine_similarity

       try:
            vec_x = vectorizer.transform([desc_x])
            vec_y = vectorizer.transform([desc_y])
            similarity = cosine_similarity(vec_x, vec_y)[0][0]

            if similarity > 0.85:
                result = "✅ Same Security"
            elif similarity > 0.5:
                result = "⚠️ Partial Match"
            else:
                result = "❌ Different Security"

            st.subheader("Prediction Result")
            st.write(f"Similarity Score: {similarity:.2f}")
            st.markdown(f"### {result}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



