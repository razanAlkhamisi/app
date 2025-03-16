import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk

nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_resource
def load_model_vectorizer():
    try:
        model = joblib.load('MultinomialNB_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        st.success("✅ Model and vectorizer loaded successfully!")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model or vectorizer file not found!")
        return None, None  

model, tfidf_vectorizer = load_model_vectorizer()

if model is None or tfidf_vectorizer is None:
    st.stop() 

class_names = ['Fake', 'Real']

st.title("📰 Fake and Real News Classifier")
st.write("🔍 Upload a CSV file with a 'text' column to classify news as Fake or Real.")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def predict_news(text):
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)
   
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    
    prediction = model.predict(vectorized_text)[0]

    return class_names[prediction]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'text' in df.columns:
        st.write("📊 Preview of the uploaded data:")
        st.write(df.head())

        df['prediction'] = df['text'].apply(predict_news)

        st.write("✅ Predictions:")
        st.write(df[['text', 'prediction']])

        st.download_button(
            label="⬇ Download Predictions as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv',
        )
    else:
        st.error("⚠ The CSV file must contain a 'text' column.")
