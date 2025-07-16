import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from text_preprocessing import clean_text

def train_default_model():
    sample_data = {
        'purpose_text': [...],
        'transaction_type': [...]
    }
    df = pd.DataFrame(sample_data)
    df['cleaned_text'] = df['purpose_text'].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['transaction_type']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'transaction_model.pkl')
    joblib.dump(vectorizer, 'transaction_vectorizer.pkl')
    return model, vectorizer

def load_model():
    model = joblib.load('transaction_model.pkl')
    vectorizer = joblib.load('transaction_vectorizer.pkl')
    return model, vectorizer

