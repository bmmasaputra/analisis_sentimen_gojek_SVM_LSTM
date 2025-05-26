import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load model dan tokenizer
@st.cache_resource
def load_models():
    svm_model = joblib.load('./Model/SVM/svm_model.pkl')
    tfidf = joblib.load('./Model/SVM/tfidf_vectorizer.pkl')
    
    lstm_model = load_model('./Model/LSTM/lstm_model.h5')
    tokenizer = joblib.load('./Model/LSTM/tokenizer.pkl')
    
    return svm_model, tfidf, lstm_model, tokenizer

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Fungsi prediksi
def predict_sentiment(text, model_choice, svm_model, tfidf, lstm_model, tokenizer):
    cleaned = preprocess_text(text)
    st.write("Cleaned Text:", cleaned)
    label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}

    if model_choice == 'SVM':
        vec = tfidf.transform([cleaned])
        probs = svm_model.predict_proba(vec)[0]
        st.write("Confidence Scores:", probs)
        pred = svm_model.predict(vec)[0]
    else:  # LSTM
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100)
        probs = lstm_model.predict(padded, verbose=0)[0]
        st.write("Confidence Scores:", probs)
        pred = label_dict[np.argmax(probs)]

    return pred, probs

# Setup page
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("🔍 Sentiment Analysis with SVM & LSTM")

# Load models
svm_model, tfidf, lstm_model, tokenizer = load_models()

# Sidebar
st.sidebar.header("⚙️ Konfigurasi Model")
model_option = st.sidebar.selectbox("Pilih model yang akan digunakan", ['SVM', 'LSTM'])

# Input box
st.subheader("✍️ Masukkan ulasan aplikasi di bawah ini:")
text_input = st.text_area("Tulis ulasan Anda di sini...", height=150)

# Action
if st.button("🔎 Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("Harap masukkan teks terlebih dahulu.")
    else:
        pred, conf_scores = predict_sentiment(text_input, model_option, svm_model, tfidf, lstm_model, tokenizer)

        label_names = ['negative', 'neutral', 'positive']
        score_df = pd.DataFrame({
            'Sentiment': label_names,
            'Confidence': conf_scores
        })

        # Layout 2 kolom: hasil & grafik
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Hasil Prediksi")
            st.markdown(f"#### Model digunakan: \n### `{model_option}`")
            st.markdown(f"#### Prediksi Sentimen: \n### `{pred}`")
            st.markdown(f"#### Confidence: \n ### `{np.max(conf_scores) * 100:.2f}%`")

        with col2:
            st.subheader("📊 Confidence Score")
            fig, ax = plt.subplots()
            sns.barplot(x='Confidence', y='Sentiment', data=score_df, palette='coolwarm', ax=ax)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_title("Confidence per Kategori Sentimen")
            st.pyplot(fig)
