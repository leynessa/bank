import os
import sys
import argparse
import pandas as pd
import joblib
from datetime import datetime
import sklearn
def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'sklearn',  # Fixed: sklearn is the import name for scikit-learn
        'nltk': 'nltk',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    for import_name, package_name in package_map.items():
        try:
            __import__(import_name)
            print(f" {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f" {package_name}")
    
    if missing_packages:
        print(f"\n Missing required packages: {missing_packages}")
        print("Please install them using:")
        if 'sklearn' in missing_packages:
            missing_packages = [p if p != 'sklearn' else 'scikit-learn' for p in missing_packages]
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print(" All dependencies are installed!")
    return True

def check_data_file():
    """Check if the data file exists"""
    data_path = 'data/dataset.csv'
    if not os.path.exists(data_path):
        print(f" Data file not found at {data_path}")
        print("Please ensure your dataset is in the correct location")
        print("Expected format: CSV with columns 'purpose_text' and 'transaction_type'")
        return False
    
    return True

def run_data_analysis():
    """Run data analysis if requested"""
    print("Running data analysis...")
    try:
        from load_real_data import main as analyze_data
        analyze_data()
        print(" Data analysis completed")
    except Exception as e:
        print(f" Data analysis failed: {e}")

def run_model_training(quick_mode=False):
    """Run model training and comparison"""
    print("Running model training and comparison...")
    
    try:
        # Import the enhanced training script
        sys.path.append('.')
        
        # Run the main training function
        if quick_mode:
            print("Running in quick mode (no hyperparameter tuning)")
            # You can modify this to skip grid search
        
        # Import and run the training script
        from train_model import main as train_models
        train_models()
        print(" Model training completed")
        
        # Load and display results
        display_training_results()
        
    except Exception as e:
        print(f" Model training failed: {e}")
        print("Please check the error messages above")

def display_training_results():
    """Display the results of model training"""
    try:
        # Check if model files exist
        model_files = ['transaction_model.pkl', 'transaction_vectorizer.pkl', 'model_metadata.pkl']
        
        for file in model_files:
            if os.path.exists(file):
                print(f" {file} created successfully")
            else:
                print(f" {file} not found")
        
        # Load and display metadata if available
        if os.path.exists('model_metadata.pkl'):
            metadata = joblib.load('model_metadata.pkl')
            print("\n📊 Best Model Information:")
            print(f"Model Name: {metadata.get('model_name', 'Unknown')}")
            print(f"Model Type: {metadata.get('model_type', 'Unknown')}")
        
    except Exception as e:
        print(f"Could not display results: {e}")

def test_trained_model():
    """Test the trained model with sample predictions"""
    print("\nTesting trained model...")
    
    try:
        # Load the trained model
        model = joblib.load('transaction_model.pkl')
        vectorizer = joblib.load('transaction_vectorizer.pkl')
        
        # Test with sample transactions
        test_transactions = [
            "grocery shopping at walmart",
            "netflix monthly subscription",
            "rent payment for apartment",
            "doctor visit copay",
            "gas station fill up",
            "restaurant dinner bill",
            "electricity bill payment"
        ]
        
        print("Sample predictions:")
        print("-" * 40)
        
        for transaction in test_transactions:
            # Clean and vectorize the text
            from text_preprocessing import clean_text
            cleaned = clean_text(transaction)
            vectorized = vectorizer.transform([cleaned])
            
            # Make prediction
            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized)[0].max()
            
            print(f"'{transaction}' → {prediction} (confidence: {confidence:.3f})")
        
        print(" Model testing completed")
        
    except Exception as e:
        print(f" Model testing failed: {e}")

def create_summary_report():
    """Create a summary report of the training process"""
    print("\n📋 Creating summary report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        f"# Transaction Classification Model Training Report\n",
        f"**Generated on:** {timestamp}\n\n",
        f"## Files Generated\n"
    ]
    
    # Check which files were created
    expected_files = [
        'transaction_model.pkl',
        'transaction_vectorizer.pkl', 
        'model_metadata.pkl',
        'model_comparison.png',
        'data_analysis_plots.png',
        'data_analysis_report.md'
    ]
    
    for file in expected_files:
        if os.path.exists(file):
            report_lines.append(f"{file}\n")
        else:
            report_lines.append(f" {file} (not found)\n")
    
    # Add model metadata if available
    if os.path.exists('model_metadata.pkl'):
        try:
            metadata = joblib.load('model_metadata.pkl')
            report_lines.append(f"\n## Best Model\n")
            report_lines.append(f"- **Model:** {metadata.get('model_name', 'Unknown')}\n")
            report_lines.append(f"- **Type:** {metadata.get('model_type', 'Unknown')}\n")
        except:
            pass
    
    report_lines.append(f"\n## Next Steps\n")
    report_lines.append(f"1. Run the API server: `python main.py`\n")
    report_lines.append(f"2. Test the API: `python test_api.py`\n")
    report_lines.append(f"3. Review the model comparison plot: `model_comparison.png`\n")
    
    # Save report
    with open('training_summary.md', 'w') as f:
        f.writelines(report_lines)
    
    print(" Summary report saved as 'training_summary.md'")

def main():
    """Main function to run the model comparison pipeline"""
    parser = argparse.ArgumentParser(description='Run transaction classification model comparison')
    parser.add_argument('--analyze-data', action='store_true', help='Run data analysis first')
    parser.add_argument('--quick', action='store_true', help='Run in quick mode (no hyperparameter tuning)')
    parser.add_argument('--test-model', action='store_true', help='Test the trained model after training')
    parser.add_argument('--skip-deps-check', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    print(" Starting Transaction Classification Model Comparison")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_deps_check:
        if not check_dependencies():
            return
    
    # Check data file
    if not check_data_file():
        return
    
    # Run data analysis if requested
    if args.analyze_data:
        run_data_analysis()
        print()
    
    # Run model training
    run_model_training(quick_mode=args.quick)
    
    # Test model if requested
    if args.test_model:
        test_trained_model()
    
    # Create summary report
    create_summary_report()
    
    print("\n Model comparison pipeline completed!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
