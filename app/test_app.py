"""
Test script to verify dashboard data loading
Run this before starting the Streamlit app
"""
import os
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """Test all data loading functions"""
    
    files_to_check = {
        "Predictions": os.path.join(BASE_DIR, "reports", "predictions.csv"),
        "Evaluation Metrics": os.path.join(BASE_DIR, "reports", "evaluation_metrics.csv"),
        "Feature Importance": os.path.join(BASE_DIR, "reports", "feature_importance.csv"),
        "Model Comparison": os.path.join(BASE_DIR, "reports", "model_comparison.csv"),
        "Featured Dataset": os.path.join(BASE_DIR, "data", "processed", "featured_dataset.csv"),
    }
    
    print("\n" + "="*70)
    print("DASHBOARD DATA LOADING TEST")
    print("="*70 + "\n")
    
    all_good = True
    
    for name, path in files_to_check.items():
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"[PASS] {name}")
                print(f"       Path: {path}")
                print(f"       Rows: {len(df)}, Columns: {len(df.columns)}")
                
                # Show column names
                cols = list(df.columns)[:5]
                print(f"       Columns: {cols}{'...' if len(df.columns) > 5 else ''}\n")
            else:
                print(f"[FAIL] {name}")
                print(f"       Path not found: {path}\n")
                all_good = False
        except Exception as e:
            print(f"[FAIL] {name}")
            print(f"       Error: {str(e)}\n")
            all_good = False
    
    print("="*70)
    if all_good:
        print("SUCCESS: All data files loaded successfully!")
        print("\nYou can now run: streamlit run app/app.py")
    else:
        print("ERROR: Some data files are missing or corrupt.")
        print("Please run the pipeline notebooks first.")
    print("="*70 + "\n")
    
    return all_good

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
