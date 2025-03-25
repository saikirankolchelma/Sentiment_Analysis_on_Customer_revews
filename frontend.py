import streamlit as st
import requests
from utils import setup_logging

logger = setup_logging()
BACKEND_URL = "http://localhost:8000"

st.title("Sentiment Analysis App")

review_text = st.text_area("Enter your review here:")
if st.button("Analyze"):
    if review_text:
        try:
            response = requests.post(f"{BACKEND_URL}/predict/", json={"review_text": review_text})
            if response.status_code == 200:
                result = response.json()
                st.write(f"**Sentiment:** {result['sentiment']}")
                st.write(f"**Confidence:** {result['confidence']:.2f}")
                logger.info(f"Frontend received: {result}")
            else:
                st.error("Error analyzing sentiment.")
                logger.error(f"API error: {response.status_code}")
        except Exception as e:
            st.error("Connection error.")
            logger.error(f"Frontend error: {e}")
    else:
        st.warning("Please enter a review.")

if st.button("Show Stored Reviews"):
    try:
        response = requests.get(f"{BACKEND_URL}/reviews/")
        if response.status_code == 200:
            reviews = response.json()
            st.table(reviews)
        else:
            st.error("Error fetching reviews.")
    except Exception as e:
        st.error("Connection error.")
        logger.error(f"Error fetching reviews: {e}")