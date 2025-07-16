# Data Processing Documentation

## Dataset Overview
- **Total transactions**: 600
- **Missing purpose_text**: 5%
- **Transaction categories**: 10
- **Final clean dataset**: 570 transactions

## Data Preprocessing Steps

### 1. Missing Data Handling
- Identify and remove 30 transactions with missing data as it would have a very small impact on the training 


### 2. Text Cleaning Process
- Convert to lowercase for consistency
- Remove punctuation except hyphens and hashtags (preserve meaningful formatting)
- Tokenize using simple space-based splitting
- Remove common stopwords
- Filter out words shorter than 3 characters
- Keep numbers as they may be meaningful 

### 3. Feature Engineering
- **Original text length**: Character count of raw text
- **Word count**: Number of words in original text  
- **Cleaned word count**: Number of words after preprocessing
- **TF-IDF features**: Vectorized text with unigrams and bigrams

### 4. Text Statistics
- Average original text length: 20.5 characters
- Average word count: 3.2 words
- Average cleaned word count: 2.8 words

## Dataset Split
- **Training set**: 60% 
- **Validation set**: 20% 
- **Test set**: 20% 


## Text Vectorization
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Parameters**:
  - Max features: 5000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Min document frequency: 2
  - Max document frequency: 0.95
  - Stop words: English stopwords removed
