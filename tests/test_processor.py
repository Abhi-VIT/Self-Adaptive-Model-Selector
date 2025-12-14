import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor, DatasetInfo

def test_binary_classification_detection():
    # Create simple binary classification dataset
    df = pd.DataFrame({
        'feat1': np.random.randn(100),
        'feat2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    processor = DataProcessor(target_col='target')
    X, y, info = processor.process_and_analyze(df)
    
    print(f"Binary Class Info: {info}")
    assert info.problem_type == "classification"
    assert info.classification_type == "binary"
    assert info.n_features == 2
    assert info.n_samples == 100

def test_multiclass_string_detection():
    # Create multiclass dataset with string labels
    df = pd.DataFrame({
        'feat1': np.random.randn(100),
        'target': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    processor = DataProcessor(target_col='target')
    X, y, info = processor.process_and_analyze(df)
    
    print(f"Multiclass Info: {info}")
    assert info.problem_type == "classification"
    assert info.classification_type == "multiclass"
    # Check encoding happened
    assert pd.api.types.is_numeric_dtype(y)

def test_regression_detection():
    # Create regression dataset
    df = pd.DataFrame({
        'feat1': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    processor = DataProcessor(target_col='target')
    X, y, info = processor.process_and_analyze(df)
    
    print(f"Regression Info: {info}")
    assert info.problem_type == "regression"
    assert info.classification_type is None

def test_time_series_detection():
    # Create time series dataset
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'val': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    processor = DataProcessor(target_col='target')
    X, y, info = processor.process_and_analyze(df)
    
    print(f"Time Series Info: {info}")
    assert info.is_time_series is True

def test_missing_imputation():
    # Create dataset with missing values
    df = pd.DataFrame({
        'num': [1.0, 2.0, np.nan, 4.0],
        'cat': ['A', 'B', np.nan, 'A'],
        'target': [1, 0, 1, 0]
    })
    
    processor = DataProcessor(target_col='target')
    X, y, info = processor.process_and_analyze(df)
    
    print(f"Missing Values Info: {info}")
    assert info.missing_percentage > 0
    assert X['num'].isnull().sum() == 0
    assert X['cat'].isnull().sum() == 0
    # Median of 1, 2, 4 is 2.0
    assert X.loc[2, 'num'] == 2.0 
    
if __name__ == "__main__":
    try:
        print("Starting test_binary_classification_detection...")
        test_binary_classification_detection()
        print("PASSED test_binary_classification_detection")
        
        print("Starting test_multiclass_string_detection...")
        test_multiclass_string_detection()
        print("PASSED test_multiclass_string_detection")
        
        print("Starting test_regression_detection...")
        test_regression_detection()
        print("PASSED test_regression_detection")
        
        print("Starting test_time_series_detection...")
        test_time_series_detection()
        print("PASSED test_time_series_detection")
        
        print("Starting test_missing_imputation...")
        test_missing_imputation()
        print("PASSED test_missing_imputation")
        
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
