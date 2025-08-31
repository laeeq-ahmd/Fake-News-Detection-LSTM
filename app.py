# app.py - Fake News Detection App

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import string
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="⚡ Fake News Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .fake-prediction {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .real-prediction {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-fill {
        height: 20px;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer with caching"""
    try:
        # Load model
        model = tf.keras.models.load_model("Models/lstm_trained.h5")
        
        # Load tokenizer
        with open("Models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def clean_text(text):
    """Text cleaning function"""
    if not text:
        return ""
    
    text = str(text).lower()
    
    # Remove URLs and HTML
    text = re.sub(r'https?://\S+|www\.\S+|<.*?>', '', text)
    
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords (only most common ones for speed)
    common_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = text.split()
    words = [word for word in words if word not in common_stopwords and len(word) > 2]
    
    return ' '.join(words).strip()

def predict_news(text, model, tokenizer):
    """Prediction function"""
    try:
        # Clean text
        cleaned_text = clean_text(text)
        
        if len(cleaned_text) < 10:
            return "INVALID", 0.0, "Text too short for analysis"
        
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded, verbose=0)[0][0]
        
        # Determine label and confidence
        if prediction >= 0.5:
            label = "REAL"
            confidence = prediction
        else:
            label = "FAKE"
            confidence = 1 - prediction
            
        return label, confidence, cleaned_text
        
    except Exception as e:
        return "ERROR", 0.0, str(e)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ Fake News Detector</h1>', unsafe_allow_html=True)
    
    # Sidebar with usage info
    with st.sidebar:
        st.header("📋 How to Use")
        
        st.markdown("**Steps:**")
        st.markdown("1. Enter news text")
        st.markdown("2. Click 'Detect'")
        st.markdown("3. Get instant results")
    
    # Load model
    with st.spinner("⚡ Loading model..."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model. Please check if 'Models/lstm_trained.h5' and 'Models/tokenizer.pkl' exist.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main content area
    st.header("📰 Enter News Text")
    
    # Text input
    input_text = st.text_area(
        "Paste any news article, headline, or text below:",
        height=200,
        placeholder="Enter news text here..."
    )
    
    # Detection button
    if st.button("🔍 Detect Fake News", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            # Show progress
            with st.spinner("🔍 Analyzing..."):
                start_time = time.time()
                
                # Get prediction
                label, confidence, cleaned_text = predict_news(input_text, model, tokenizer)
                
                end_time = time.time()
                prediction_time = end_time - start_time
            
            # Display results
            st.markdown("---")
            st.header("🎯 Detection Results")
            
            # Prediction box
            if label == "FAKE":
                st.markdown(f'<div class="prediction-box fake-prediction">🚨 FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
            elif label == "REAL":
                st.markdown(f'<div class="prediction-box real-prediction">✅ REAL NEWS DETECTED</div>', unsafe_allow_html=True)
            else:
                st.error(f"❌ Error: {cleaned_text}")
                return
            
            # Confidence bar
            st.markdown("**Confidence Level:**")
            confidence_percent = confidence * 100
            
            # Color based on confidence
            if confidence_percent >= 80:
                color = "#4caf50"  # Green
            elif confidence_percent >= 60:
                color = "#ff9800"  # Orange
            else:
                color = "#f44336"  # Red
            
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {color};"></div>
            </div>
            <p style="text-align: center; font-weight: bold;">{confidence_percent:.1f}%</p>
            """, unsafe_allow_html=True)
            
            # Additional info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Prediction Time", f"{prediction_time:.3f}s")
            with col_info2:
                st.metric("Text Length", f"{len(input_text)} chars")
            
            # Show cleaned text
            with st.expander("🔍 View Processed Text"):
                st.text(cleaned_text)

if __name__ == "__main__":
    main() 