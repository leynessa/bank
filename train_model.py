import pandas as pd
import numpy as np
import re
import joblib
import nltk
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"NLTK download failed: {e}")

download_nltk_data()

# Initialize NLP tools
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    print(f"NLTK initialization failed: {e}")
    stop_words = set()
    lemmatizer = None

def clean_text(text: str) -> str:
    """Clean and preprocess text data"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # If NLTK is available, use it
    if lemmatizer is not None:
        try:
            words = word_tokenize(text)
            words = [lemmatizer.lemmatize(word) for word in words 
                    if word not in stop_words and len(word) > 2]
            return ' '.join(words)
        except:
            pass
    
    # Fallback: basic word processing
    words = text.split()
    words = [w for w in words if len(w) > 2]
    return ' '.join(words)

def load_dataset():
    """Load the actual dataset from CSV"""
    try:
        df = pd.read_csv('data/dataset.csv')
        print(f"Dataset loaded with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print("Dataset not found at 'data/dataset.csv'. Using sample data instead.")
        return get_sample_data()

def get_sample_data():
    """Fallback sample data based on your dataset sample"""
    return pd.DataFrame({
        'purpose_text': [
            'Mini golf - processed', 'Mini golf #4760', 'Rent - Main St Apt - pending',
            'medical insurance', 'rent for apartment', 'MCDONALD\'S MEAL',
            'grocery store shopping', 'supermarket weekly groceries', 'food shopping',
            'restaurant dinner', 'fast food lunch', 'coffee shop',
            'gas station fuel', 'uber ride', 'taxi fare', 'bus ticket',
            'netflix subscription', 'spotify premium', 'gym membership',
            'doctor appointment', 'pharmacy prescription', 'dental checkup',
            'amazon purchase', 'online shopping', 'electronics store',
            'monthly rent payment', 'apartment rent', 'house rent',
            'electricity bill', 'water bill', 'internet service',
            'movie theater', 'concert tickets', 'amusement park'
        ],
        'transaction_type': [
            'entertainment', 'entertainment', 'rent',
            'healthcare', 'rent', 'dining',
            'groceries', 'groceries', 'groceries',
            'dining', 'dining', 'dining',
            'transportation', 'transportation', 'transportation', 'transportation',
            'subscription', 'subscription', 'subscription',
            'healthcare', 'healthcare', 'healthcare',
            'shopping', 'shopping', 'shopping',
            'rent', 'rent', 'rent',
            'utilities', 'utilities', 'utilities',
            'entertainment', 'entertainment', 'entertainment'
        ]
    })

def prepare_data_for_training():
    """Prepare the dataset for model training"""
    df = load_dataset()
    
    # Check if required columns exist
    required_columns = ['purpose_text', 'transaction_type']
    if not all(col in df.columns for col in required_columns):
        print(f"Required columns {required_columns} not found in dataset")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Clean the data
    print(f"Original dataset size: {len(df)}")
    df = df.dropna(subset=['purpose_text', 'transaction_type'])
    print(f"After removing missing values: {len(df)}")
    
    # Clean text
    df['cleaned_text'] = df['purpose_text'].apply(clean_text)
    
    # Remove empty cleaned text
    df = df[df['cleaned_text'].str.strip() != '']
    print(f"After text cleaning: {len(df)}")
    
    # Display class distribution
    print("\nClass distribution:")
    print(df['transaction_type'].value_counts())
    
    return df

def get_model_configurations():
    """Define models with their hyperparameters for grid search"""
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Naive Bayes': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }
    return models

def train_and_evaluate_models(X_train, X_test, y_train, y_test, use_grid_search=True):
    """Train and evaluate multiple models"""
    models = get_model_configurations()
    results = {}
    trained_models = {}
    
    print("Training and evaluating models...")
    print("=" * 50)
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        if use_grid_search and config['params']:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'], 
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            best_model = config['model']
            best_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred
        }
        
        trained_models[name] = best_model
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    return results, trained_models

def display_results(results, y_test):
    """Display model comparison results"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'CV Mean': [results[model]['cv_mean'] for model in results.keys()],
        'CV Std': [results[model]['cv_std'] for model in results.keys()]
    })
    
    # Sort by test accuracy
    results_df = results_df.sort_values('Test Accuracy', ascending=False)
    
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Find best model
    best_model_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Test Accuracy']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    
    # Detailed classification report for best model
    print(f"\nDetailed Classification Report for {best_model_name}:")
    print("-" * 50)
    y_pred_best = results[best_model_name]['predictions']
    print(classification_report(y_test, y_pred_best))
    
    return best_model_name, results_df

def plot_model_comparison(results_df):
    """Plot model comparison results"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Test Accuracy
        plt.subplot(1, 2, 1)
        plt.barh(results_df['Model'], results_df['Test Accuracy'])
        plt.xlabel('Test Accuracy')
        plt.title('Model Comparison - Test Accuracy')
        plt.xlim(0, 1)
        
        # Plot 2: CV scores with error bars
        plt.subplot(1, 2, 2)
        plt.errorbar(results_df['CV Mean'], results_df['Model'], 
                    xerr=results_df['CV Std'], fmt='o', capsize=5)
        plt.xlabel('Cross-Validation Score')
        plt.title('Model Comparison - CV Score (with std)')
        plt.xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Model comparison plot saved as 'model_comparison.png'")
    except Exception as e:
        print(f"Could not create plots: {e}")

def save_best_model(best_model_name, trained_models, vectorizer):
    """Save the best model and vectorizer"""
    best_model = trained_models[best_model_name]
    
    # Save model and vectorizer
    joblib.dump(best_model, 'transaction_model.pkl')
    joblib.dump(vectorizer, 'transaction_vectorizer.pkl')
    
    # Save model metadata
    metadata = {
        'model_name': best_model_name,
        'model_type': type(best_model).__name__,
        'vectorizer_params': vectorizer.get_params(),
        'model_params': best_model.get_params()
    }
    
    joblib.dump(metadata, 'model_metadata.pkl')
    
    print(f"\nBest model ({best_model_name}) saved successfully!")
    print("Files saved:")
    print("- transaction_model.pkl")
    print("- transaction_vectorizer.pkl")
    print("- model_metadata.pkl")

def main():
    """Main training pipeline"""
    print("Starting Multi-Model Training Pipeline")
    print("=" * 50)
    
    # Load and prepare data
    df = prepare_data_for_training()
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"\nTraining with {len(df)} samples")
    print(f"Transaction types: {sorted(df['transaction_type'].unique())}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['transaction_type'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['transaction_type']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize text
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature vector shape: {X_train_vec.shape}")
    
    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(
        X_train_vec, X_test_vec, y_train, y_test, use_grid_search=True
    )
    
    # Display results
    best_model_name, results_df = display_results(results, y_test)
    
    # Plot comparison
    plot_model_comparison(results_df)
    
    # Save best model
    save_best_model(best_model_name, trained_models, vectorizer)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()