# LLM Approach for Transaction Classification

## Conceptual Approach

### 1. Model Selection
- **BERT-based models**: Good for understanding context in short texts
- **DistilBERT**: Lighter version, faster training
- **RoBERTa**: Often better performance than BERT

### 2. Fine-tuning Process
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(unique_labels)
)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['purpose_text'], truncation=True, padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)