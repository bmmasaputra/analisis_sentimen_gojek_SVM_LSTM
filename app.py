import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load new models and components
@st.cache_resource
def load_models():
    try:
        # Load SVM model and TF-IDF vectorizer (new models)
        svm_model = joblib.load('./Improved Model/SVM/svm_sentiment_model.pkl')
        tfidf = joblib.load('./Improved Model/SVM/tfidf_vectorizer.pkl')
        
        # Load LSTM model (new model)
        lstm_model = load_model('./Improved Model/LSTM/lstm_sentiment_model.h5')
        
        # Load tokenizer for LSTM (new tokenizer)
        with open('./Improved Model/LSTM/lstm_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load label encoder (new component)
        with open('./Improved Model/LSTM/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Load model configuration (new component)
        with open('./Improved Model/LSTM/model_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        st.success("✅ All models loaded successfully!")
        return svm_model, tfidf, lstm_model, tokenizer, label_encoder, config
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.error("Make sure these files are in the root directory:")
        st.error("- svm_sentiment_model.pkl")
        st.error("- tfidf_vectorizer.pkl") 
        st.error("- lstm_sentiment_model.h5")
        st.error("- lstm_tokenizer.pkl")
        st.error("- label_encoder.pkl")
        st.error("- model_config.pkl")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None, None

# Enhanced preprocessing function
def preprocess_text(text):
    """Enhanced text preprocessing for Indonesian text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#\w+','', text)
    
    # Keep Indonesian characters and basic punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Enhanced prediction function
def predict_sentiment(text, model_choice, svm_model, tfidf, lstm_model, tokenizer, label_encoder, config):
    """Predict sentiment using the new trained models"""
    
    # Preprocess text
    cleaned = preprocess_text(text)
    
    # Display cleaned text
    with st.expander("🔍 View Preprocessed Text"):
        st.write("**Original Text:**", text)
        st.write("**Cleaned Text:**", cleaned)
    
    if model_choice == 'SVM':
        try:
            # Transform text using TF-IDF
            vec = tfidf.transform([cleaned])
            
            # Get prediction probabilities
            try:
                probs = svm_model.predict_proba(vec)[0]
                has_proba = True
            except:
                # If predict_proba is not available, use decision_function
                decision_scores = svm_model.decision_function(vec)[0]
                # Convert decision scores to probabilities (approximation)
                if len(decision_scores.shape) == 0:  # Binary classification
                    decision_scores = np.array([decision_scores])
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                probs = exp_scores / np.sum(exp_scores)
                has_proba = False
            
            # Get prediction
            pred_idx = svm_model.predict(vec)[0]
            pred = label_encoder.inverse_transform([pred_idx])[0]
            
            # Create confidence scores for all classes
            if len(probs) != len(label_encoder.classes_):
                # Handle case where probabilities don't match number of classes
                full_probs = np.zeros(len(label_encoder.classes_))
                full_probs[pred_idx] = np.max(probs) if len(probs) > 0 else 1.0
                probs = full_probs
            
        except Exception as e:
            st.error(f"SVM Prediction Error: {e}")
            return None, None
            
    else:  # LSTM
        try:
            # Tokenize and pad text
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=config['max_length'], padding='post')
            
            # Get prediction probabilities
            probs = lstm_model.predict(padded, verbose=0)[0]
            
            # Get prediction
            pred_idx = np.argmax(probs)
            pred = label_encoder.inverse_transform([pred_idx])[0]
            
        except Exception as e:
            st.error(f"LSTM Prediction Error: {e}")
            return None, None

    return pred, probs

# Setup page configuration
st.set_page_config(
    page_title="Sentiment Analyzer - Gojek Reviews", 
    layout="wide",
    page_icon="🔍"
)

# Header
st.title("🔍Gojek Sentiment Analysis with SVM & LSTM")
st.markdown("### Analyze Indonesian App Reviews with Advanced NLP Models")

# Load models
with st.spinner("Loading models..."):
    svm_model, tfidf, lstm_model, tokenizer, label_encoder, config = load_models()

# Check if models loaded successfully
if svm_model is None:
    st.stop()

# Display model information
with st.expander("ℹ️ Model Information"):
    st.write("**Available Models:**")
    st.write("- **SVM with TF-IDF**: Traditional machine learning approach")
    st.write("- **LSTM with Word2Vec**: Deep learning approach")
    st.write(f"**Available Sentiments:** {', '.join(label_encoder.classes_)}")
    st.write(f"**Max Sequence Length (LSTM):** {config['max_length']}")

# Sidebar configuration
st.sidebar.header("⚙️ Model Configuration")
model_option = st.sidebar.selectbox(
    "Choose the model to use:", 
    ['SVM', 'LSTM'],
    help="SVM: Fast and efficient. LSTM: Better for context understanding"
)

# Model comparison option
show_comparison = st.sidebar.checkbox(
    "🔄 Compare Both Models", 
    help="Show predictions from both SVM and LSTM models"
)

# Main input section
st.subheader("✍️ Enter your app review below:")
text_input = st.text_area(
    "Write your review here... (Indonesian language recommended)",
    height=150,
    placeholder="Example: Aplikasi sangat bagus dan mudah digunakan!"
)

# Sample reviews for quick testing
st.subheader("🎯 Quick Test with Sample Reviews:")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Positive Sample"):
        text_input = "Aplikasi gojek sangat membantu dan mudah digunakan, driver ramah"

with col2:
    if st.button("😐 Neutral Sample"):
        text_input = "Aplikasi biasa saja tidak ada yang istimewa"

with col3:
    if st.button("😞 Negative Sample"):
        text_input = "Aplikasi sering error dan lambat sekali"

# Prediction section
if st.button("🔎 Predict Sentiment", type="primary"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        if show_comparison:
            # Compare both models
            st.subheader("🔄 Model Comparison")
            
            col1, col2 = st.columns(2)
            
            # SVM Results
            with col1:
                st.markdown("#### 🤖 SVM Model")
                svm_pred, svm_conf = predict_sentiment(
                    text_input, 'SVM', svm_model, tfidf, lstm_model, 
                    tokenizer, label_encoder, config
                )
                
                if svm_pred is not None:
                    # Determine color based on sentiment
                    color = "🟢" if svm_pred == "positif" else "🔴" if svm_pred == "negatif" else "🟡"
                    
                    st.markdown(f"**Prediction:** {color} **{svm_pred.upper()}**")
                    st.markdown(f"**Confidence:** {np.max(svm_conf) * 100:.2f}%")
                    
                    # Confidence breakdown
                    svm_df = pd.DataFrame({
                        'Sentiment': label_encoder.classes_,
                        'Confidence': svm_conf
                    })
                    
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=svm_df, x='Confidence', y='Sentiment', 
                               palette='viridis', ax=ax1)
                    ax1.set_xlim(0, 1)
                    ax1.set_title("SVM Confidence Scores")
                    st.pyplot(fig1)
            
            # LSTM Results
            with col2:
                st.markdown("#### 🧠 LSTM Model")
                lstm_pred, lstm_conf = predict_sentiment(
                    text_input, 'LSTM', svm_model, tfidf, lstm_model, 
                    tokenizer, label_encoder, config
                )
                
                if lstm_pred is not None:
                    # Determine color based on sentiment
                    color = "🟢" if lstm_pred == "positif" else "🔴" if lstm_pred == "negatif" else "🟡"
                    
                    st.markdown(f"**Prediction:** {color} **{lstm_pred.upper()}**")
                    st.markdown(f"**Confidence:** {np.max(lstm_conf) * 100:.2f}%")
                    
                    # Confidence breakdown
                    lstm_df = pd.DataFrame({
                        'Sentiment': label_encoder.classes_,
                        'Confidence': lstm_conf
                    })
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    sns.barplot(data=lstm_df, x='Confidence', y='Sentiment', 
                               palette='plasma', ax=ax2)
                    ax2.set_xlim(0, 1)
                    ax2.set_title("LSTM Confidence Scores")
                    st.pyplot(fig2)
            
            # Agreement analysis
            if svm_pred and lstm_pred:
                st.subheader("🤝 Model Agreement")
                if svm_pred == lstm_pred:
                    st.success(f"✅ Both models agree: **{svm_pred.upper()}**")
                else:
                    st.warning(f"❌ Models disagree: SVM says **{svm_pred}**, LSTM says **{lstm_pred}**")
        
        else:
            # Single model prediction
            pred, conf_scores = predict_sentiment(
                text_input, model_option, svm_model, tfidf, lstm_model, 
                tokenizer, label_encoder, config
            )
            
            if pred is not None:
                # Create results layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📌 Prediction Results")
                    
                    # Determine emoji and color based on sentiment
                    if pred == "positif":
                        emoji = "😊"
                        color = "green"
                    elif pred == "negatif":
                        emoji = "😞" 
                        color = "red"
                    else:
                        emoji = "😐"
                        color = "orange"
                    
                    st.markdown(f"**Model Used:** `{model_option}`")
                    st.markdown(f"**Sentiment:** {emoji} **:{color}[{pred.upper()}]**")
                    st.markdown(f"**Confidence:** **{np.max(conf_scores) * 100:.2f}%**")
                    
                    # Confidence breakdown table
                    conf_df = pd.DataFrame({
                        'Sentiment': label_encoder.classes_,
                        'Confidence': conf_scores,
                        'Percentage': [f"{score*100:.1f}%" for score in conf_scores]
                    })
                    st.dataframe(conf_df, hide_index=True)
                
                with col2:
                    st.subheader("📊 Confidence Visualization")
                    
                    # Create confidence chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Color mapping for sentiments
                    colors = []
                    for sentiment in label_encoder.classes_:
                        if sentiment == "positif":
                            colors.append("#2E8B57")  # Green
                        elif sentiment == "negatif":
                            colors.append("#DC143C")  # Red
                        else:
                            colors.append("#FF8C00")  # Orange
                    
                    bars = sns.barplot(
                        x=conf_scores, 
                        y=label_encoder.classes_, 
                        palette=colors, 
                        ax=ax
                    )
                    
                    # Customize chart
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Confidence Score")
                    ax.set_ylabel("Sentiment")
                    ax.set_title(f"{model_option} Model - Confidence Distribution")
                    
                    # Add percentage labels on bars
                    for i, (bar, score) in enumerate(zip(bars.patches, conf_scores)):
                        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{score*100:.1f}%', 
                               va='center', fontweight='bold')
                    
                    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Built with Streamlit | Powered by SVM & LSTM Models**")

# Additional features in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 About the Models")
st.sidebar.write("**SVM (Support Vector Machine):**")
st.sidebar.write("- Uses TF-IDF features")
st.sidebar.write("- Fast prediction")
st.sidebar.write("- Good for traditional text analysis")

st.sidebar.write("**LSTM (Long Short-Term Memory):**")
st.sidebar.write("- Uses Word2Vec embeddings")
st.sidebar.write("- Understands context better")
st.sidebar.write("- Deep learning approach")

# Batch processing option
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Batch Processing")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write(f"Loaded {len(df)} rows")
        
        # Select text column
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        selected_column = st.sidebar.selectbox("Select text column:", text_columns)
        
        if st.sidebar.button("Process Batch"):
            with st.spinner("Processing batch predictions..."):
                results = []
                
                for text in df[selected_column].astype(str):
                    pred, conf = predict_sentiment(
                        text, model_option, svm_model, tfidf, lstm_model,
                        tokenizer, label_encoder, config
                    )
                    results.append({
                        'text': text,
                        'prediction': pred if pred else 'error',
                        'confidence': np.max(conf) if conf is not None else 0
                    })
                
                results_df = pd.DataFrame(results)
                st.subheader("📊 Batch Processing Results")
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")