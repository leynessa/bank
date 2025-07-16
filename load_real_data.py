import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from text_preprocessing import clean_text

def load_and_analyze_dataset():
    """Load dataset and perform comprehensive analysis"""
    try:
        # Try to load the actual dataset
        df = pd.read_csv('data/dataset.csv')
        print(f"✅ Dataset loaded successfully with {len(df)} rows")
    except FileNotFoundError:
        print("❌ Dataset not found at 'data/dataset.csv'")
        print("Please ensure your dataset is in the correct location")
        return None
    
    # Display basic info
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.2f} KB")
    
    # Check for required columns
    required_columns = ['purpose_text', 'transaction_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    print("\n" + "="*50)
    print("MISSING DATA ANALYSIS")
    print("="*50)
    
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percentage
    })
    
    print(missing_df)
    
    # Check for empty strings in text columns
    if 'purpose_text' in df.columns:
        empty_text = df['purpose_text'].isna() | (df['purpose_text'].str.strip() == '')
        print(f"\nEmpty purpose_text entries: {empty_text.sum()} ({empty_text.sum()/len(df)*100:.2f}%)")
    
    return missing_df

def analyze_text_data(df):
    """Analyze text characteristics"""
    print("\n" + "="*50)
    print("TEXT DATA ANALYSIS")
    print("="*50)
    
    # Basic text statistics
    df['text_length'] = df['purpose_text'].str.len()
    df['word_count'] = df['purpose_text'].str.split().str.len()
    df['cleaned_text'] = df['purpose_text'].apply(clean_text)
    df['cleaned_word_count'] = df['cleaned_text'].str.split().str.len()
    
    text_stats = {
        'Average text length': df['text_length'].mean(),
        'Average word count': df['word_count'].mean(),
        'Average cleaned word count': df['cleaned_word_count'].mean(),
        'Min text length': df['text_length'].min(),
        'Max text length': df['text_length'].max(),
        'Min word count': df['word_count'].min(),
        'Max word count': df['word_count'].max()
    }
    
    for stat, value in text_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Show examples of different text lengths
    print("\nText length examples:")
    print("Shortest text:", df.loc[df['text_length'].idxmin(), 'purpose_text'])
    print("Longest text:", df.loc[df['text_length'].idxmax(), 'purpose_text'])
    
    return text_stats

def analyze_transaction_types(df):
    """Analyze transaction type distribution"""
    print("\n" + "="*50)
    print("TRANSACTION TYPE ANALYSIS")
    print("="*50)
    
    # Class distribution
    class_counts = df['transaction_type'].value_counts()
    class_percentages = (class_counts / len(df)) * 100
    
    class_df = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_percentages
    })
    
    print("Class distribution:")
    print(class_df)
    
    # Check for class imbalance
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 5:
        print("⚠️  High class imbalance detected!")
    
    return class_df

def analyze_text_by_category(df):
    """Analyze text patterns by transaction category"""
    print("\n" + "="*50)
    print("TEXT PATTERNS BY CATEGORY")
    print("="*50)
    
    # Average text length by category
    category_stats = df.groupby('transaction_type').agg({
        'text_length': ['mean', 'std'],
        'word_count': ['mean', 'std']
    }).round(2)
    
    print("Text statistics by category:")
    print(category_stats)
    
    # Most common words in each category
    print("\nMost common words by category:")
    for category in df['transaction_type'].unique():
        category_text = df[df['transaction_type'] == category]['cleaned_text']
        all_words = ' '.join(category_text.dropna()).split()
        common_words = Counter(all_words).most_common(5)
        print(f"{category}: {[word for word, count in common_words]}")

def create_visualizations(df):
    """Create visualizations for the dataset"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Transaction type distribution
        ax1 = axes[0, 0]
        class_counts = df['transaction_type'].value_counts()
        ax1.bar(class_counts.index, class_counts.values)
        ax1.set_title('Transaction Type Distribution')
        ax1.set_xlabel('Transaction Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Text length distribution
        ax2 = axes[0, 1]
        ax2.hist(df['text_length'], bins=20, alpha=0.7)
        ax2.set_title('Text Length Distribution')
        ax2.set_xlabel('Text Length (characters)')
        ax2.set_ylabel('Frequency')
        
        # 3. Word count distribution
        ax3 = axes[1, 0]
        ax3.hist(df['word_count'], bins=20, alpha=0.7)
        ax3.set_title('Word Count Distribution')
        ax3.set_xlabel('Word Count')
        ax3.set_ylabel('Frequency')
        
        # 4. Text length by category (boxplot)
        ax4 = axes[1, 1]
        df.boxplot(column='text_length', by='transaction_type', ax=ax4)
        ax4.set_title('Text Length by Transaction Type')
        ax4.set_xlabel('Transaction Type')
        ax4.set_ylabel('Text Length')
        
        plt.tight_layout()
        plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Visualizations saved as 'data_analysis_plots.png'")
        
    except Exception as e:
        print(f"❌ Could not create visualizations: {e}")

def generate_data_report(df):
    """Generate a comprehensive data report"""
    print("\n" + "="*50)
    print("GENERATING DATA REPORT")
    print("="*50)
    
    report = []
    report.append("# Data Analysis Report\n")
    report.append(f"**Dataset Size:** {len(df)} transactions\n")
    report.append(f"**Number of Categories:** {df['transaction_type'].nunique()}\n")
    report.append(f"**Average Text Length:** {df['purpose_text'].str.len().mean():.1f} characters\n")
    report.append(f"**Average Word Count:** {df['purpose_text'].str.split().str.len().mean():.1f} words\n")
    
    report.append("\n## Class Distribution\n")
    class_counts = df['transaction_type'].value_counts()
    for category, count in class_counts.items():
        percentage = (count / len(df)) * 100
        report.append(f"- **{category}:** {count} ({percentage:.1f}%)\n")
    
    report.append("\n## Data Quality Issues\n")
    missing_text = df['purpose_text'].isna().sum()
    if missing_text > 0:
        report.append(f"- Missing purpose_text: {missing_text} entries\n")
    
    empty_text = (df['purpose_text'].str.strip() == '').sum()
    if empty_text > 0:
        report.append(f"- Empty purpose_text: {empty_text} entries\n")
    
    if missing_text == 0 and empty_text == 0:
        report.append("- No missing or empty text entries found ✅\n")
    
    report.append("\n## Recommendations\n")
    
    # Class imbalance check
    max_class = class_counts.max()
    min_class = class_counts.min()
    if max_class / min_class > 5:
        report.append("- Consider addressing class imbalance using techniques like SMOTE or class weights\n")
    
    # Text length recommendations
    avg_length = df['purpose_text'].str.len().mean()
    if avg_length < 10:
        report.append("- Very short text entries may require specialized preprocessing\n")
    
    # Save report to file
    with open('data_analysis_report.md', 'w') as f:
        f.writelines(report)
    
    print("✅ Data analysis report saved as 'data_analysis_report.md'")
    
    return ''.join(report)

def main():
    """Main data analysis pipeline"""
    print("Starting Data Analysis Pipeline")
    print("="*50)
    
    # Load dataset
    df = load_and_analyze_dataset()
    if df is None:
        return
    
    # Perform analyses
    missing_info = analyze_missing_data(df)
    text_stats = analyze_text_data(df)
    class_info = analyze_transaction_types(df)
    analyze_text_by_category(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Generate report
    report = generate_data_report(df)
    
    print("\n" + "="*50)
    print("DATA ANALYSIS COMPLETED")
    print("="*50)
    print("Files generated:")
    print("- data_analysis_plots.png")
    print("- data_analysis_report.md")
    
    # Clean dataset for training
    df_clean = df.dropna(subset=['purpose_text', 'transaction_type'])
    df_clean = df_clean[df_clean['purpose_text'].str.strip() != '']
    
    print(f"\nCleaned dataset: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    
    # Save cleaned dataset
    df_clean[['purpose_text', 'transaction_type']].to_csv('data/cleaned_dataset.csv', index=False)
    print("✅ Cleaned dataset saved as 'data/cleaned_dataset.csv'")

if __name__ == "__main__":
    main()