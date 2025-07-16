import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# NLP and preprocessing
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# Setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data with error handling
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}")

download_nltk_data()

# Initialize NLP tools with error handling
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.warning(f"NLTK initialization failed: {e}")
    # Fallback: use basic preprocessing without NLTK
    stop_words = set()
    lemmatizer = None


# FastAPI Setup

app = FastAPI(
    title="Transaction Classification API",
    description="Classifies transaction purpose text into categories",
    version="1.0.0"
)

# Global variables
model = None
vectorizer = None


# Pydantic Models

class TransactionRequest(BaseModel):
    purpose_text: str

class TransactionResponse(BaseModel):
    predicted_type: str
    confidence: float

class BatchTransactionRequest(BaseModel):
    transactions: List[str]

class BatchTransactionResponse(BaseModel):
    predictions: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool


# Text Cleaning Function with Fallback

def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""

    text = re.sub(r"[^\w\s]", " ", text.lower())
    
    # If NLTK is available, use it
    if lemmatizer is not None:
        try:
            words = word_tokenize(text)
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
            return " ".join(words)
        except:
            pass
    
    # Fallback: basic word processing
    words = text.split()
    words = [w for w in words if len(w) > 2]
    return " ".join(words)


# Model Load / Train

def load_model():
    global model, vectorizer
    try:
        model = joblib.load("transaction_model.pkl")
        vectorizer = joblib.load("transaction_vectorizer.pkl")
        logger.info("Model and vectorizer loaded successfully.")
        return True
    except Exception as e:
        logger.warning(f"Model loading failed: {e}")
        return False

def train_default_model():
    global model, vectorizer

    # Sample data
    sample_data = {
        'purpose_text': [
            'grocery store shopping', 'supermarket weekly groceries', 'food shopping',
            'restaurant dinner', 'fast food lunch', 'coffee shop',
            'gas station fuel', 'uber ride', 'taxi fare', 'bus ticket',
            'netflix subscription', 'spotify premium', 'gym membership',
            'doctor appointment', 'pharmacy prescription', 'dental checkup',
            'amazon purchase', 'online shopping', 'electronics store',
            'monthly rent payment', 'apartment rent', 'house rent',
            'electricity bill', 'water bill', 'internet service'
        ],
        'transaction_type': [
            'groceries', 'groceries', 'groceries',
            'dining', 'dining', 'dining',
            'transportation', 'transportation', 'transportation', 'transportation',
            'subscription', 'subscription', 'subscription',
            'healthcare', 'healthcare', 'healthcare',
            'shopping', 'shopping', 'shopping',
            'rent', 'rent', 'rent',
            'utilities', 'utilities', 'utilities'
        ]
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
    logger.info("Default model trained and saved.")

# FastAPI Startup Event

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    if not load_model():
        logger.info("Training default model...")
        train_default_model()


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        message="API is running",
        model_loaded=(model is not None and vectorizer is not None)
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        message="Model loaded" if model else "Model not loaded",
        model_loaded=(model is not None and vectorizer is not None)
    )

@app.post("/classify", response_model=TransactionResponse)
async def classify_transaction(request: TransactionRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleaned = clean_text(request.purpose_text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    pred = model.classes_[np.argmax(probs)]
    confidence = round(np.max(probs), 3)

    return TransactionResponse(predicted_type=pred, confidence=confidence)

@app.post("/classify_batch", response_model=BatchTransactionResponse)
async def classify_batch(request: BatchTransactionRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    cleaned = [clean_text(t) for t in request.transactions]
    vec = vectorizer.transform(cleaned)
    preds = model.predict(vec)
    probs = model.predict_proba(vec)

    results = []
    for i in range(len(preds)):
        confidence = round(np.max(probs[i]), 3)
        results.append({
            "input": request.transactions[i],
            "predicted_type": preds[i],
            "confidence": confidence
        })

    return BatchTransactionResponse(predictions=results)


# Run the App 

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

