# cell 1: imports
!pip install sastrawi tensorflow gensim scikit-learn

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# cell 2: load and prepare data
df = pd.read_csv('GojekAppReview_1.csv')

# Sentiment labeling
def label_sentiment(score):
    if score <= 2: return 'negative'
    elif score == 3: return 'neutral'
    else: return 'positive'

df['sentiment'] = df['score'].apply(label_sentiment)

# cell 3: text preprocessing
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

normalization_dict = {
    # Basic conversions
    "gk": "gak", "ga": "gak", "tdk": "tidak", "bgt": "banget", "dr": "dari",
    "udh": "sudah", "jg": "juga", "aja": "saja", "sy": "saya", "trs": "terus",
    "ngga": "tidak", "nggak": "tidak", "bkin": "bikin", "blm": "belum",
    "sm": "sama", "tp": "tapi", "dgn": "dengan", "krn": "karena",
    
    # Additional common Indonesian slang words
    "yg": "yang", "utk": "untuk", "dg": "dengan", "klo": "kalau", "kok": "kok",
    "gw": "saya", "gue": "saya", "lu": "kamu", "kyk": "seperti", "gmn": "bagaimana",
    "sih": "sih", "deh": "deh", "dpt": "dapat", "bs": "bisa", "sdh": "sudah",
    "ttg": "tentang", "dlm": "dalam", "kl": "kalau", "km": "kamu", "hrs": "harus",
    "mk": "maka", "scr": "secara", "spy": "supaya", "bnyk": "banyak", "slh": "salah",
    "krna": "karena", "mw": "mau", "pk": "pakai", "pke": "pakai", "tq": "terima kasih",
    "thx": "terima kasih", "gpp": "tidak apa-apa", "gampng": "gampang", "bwt": "buat",
    "skrng": "sekarang", "skrg": "sekarang", "msh": "masih", "bnr": "benar",
    "trims": "terima kasih", "gk": "tidak", "gak": "tidak", "udah": "sudah",
    "pgen": "ingin", "pgn": "ingin", "kyk": "seperti", "gitu": "begitu",
    "gini": "begini", "gmana": "bagaimana", "gimana": "bagaimana", "gt": "begitu",
    "yah": "ya", "karna": "karena", "dri": "dari", "tdk": "tidak",
    "knp": "kenapa", "kpn": "kapan", "nih": "ini", "spt": "seperti",
    "ntaps": "mantap", "mantul": "mantap", "mantap": "mantap"
}

def preprocess_text(text):
    text = re.sub(r'[^\\w\\s]', '', str(text).lower().strip())
    words = text.split()
    normalized_words = [normalization_dict.get(word, word) for word in words]
    text = ' '.join(normalized_words)
    text = stemmer.stem(text)
    text = stopword.remove(text)
    return text

df['clean_content'] = df['content'].apply(preprocess_text)

# cell 4: train-test split
X = df['clean_content']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cell 5: traditional ML model (SVM with TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

y_pred_svm = svm_model.predict(X_test_tfidf)
print("SVM Performance:")
print(classification_report(y_test, y_pred_svm))

# Save traditional model
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# cell 6: deep learning model (LSTM with Word2Vec)
# Word2Vec embeddings
sentences = [text.split() for text in X_train]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Label encoding
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
y_train_num = y_train.map(label_dict)
y_test_num = y_test.map(label_dict)

# LSTM model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# Train model
history = model.fit(X_train_pad, y_train_num,
                    epochs=10,
                    batch_size=64,
                    validation_data=(X_test_pad, y_test_num))

# Evaluate
y_pred_lstm = model.predict(X_test_pad)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
print("LSTM Performance:")
print(classification_report(y_test_num, y_pred_lstm))

# Save deep learning model
model.save('lstm_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')