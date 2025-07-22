import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

# Load models and vectorizers
kmeans_model = joblib.load("kmeans_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
lda_model = joblib.load("lda_model.pkl")
count_vectorizer = joblib.load("count_vectorizer.pkl")

# Streamlit app UI
st.set_page_config(page_title="Topic Modeling App", layout="centered")
st.title("🔍 Topic Modeling with KMeans and LDA")

text_input = st.text_area("Enter a paragraph or document:", height=200)

if st.button("Predict Topic"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # KMeans Prediction
        tfidf_text = tfidf_vectorizer.transform([text_input])
        kmeans_cluster = kmeans_model.predict(tfidf_text)[0]

        # LDA Prediction
        count_text = count_vectorizer.transform([text_input])
        lda_topic_distribution = lda_model.transform(count_text)
        lda_topic = np.argmax(lda_topic_distribution)

        st.success("🌍 Results")
        st.write(f"**KMeans Cluster:** {kmeans_cluster}")
        st.write(f"**LDA Topic:** {lda_topic}")

        # Show topic distribution for LDA
        st.subheader("📊 LDA Topic Distribution:")
        for i, prob in enumerate(lda_topic_distribution[0]):
            st.write(f"Topic {i}: {prob:.4f}")

st.markdown("---")
st.markdown("Made by Eshanika Marwaha")

