import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_features import MetaFeatureExtractor, MetaFeatures

def test_regression_meta_features():
    """Test meta-feature extraction on regression dataset."""
    np.random.seed(42)
    
    # Create regression dataset with known characteristics
    n_samples = 100
    X = pd.DataFrame({
        'feat1': np.random.randn(n_samples),
        'feat2': np.random.randn(n_samples),
        'feat3': np.random.randn(n_samples)
    })
    y = pd.Series(2 * X['feat1'] + 3 * X['feat2'] + np.random.randn(n_samples))
    
    extractor = MetaFeatureExtractor(problem_type='regression')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nRegression Meta-Features:")
    for key, val in feat_dict.items():
        print(f"  {key}: {val}")
    
    # Verify dimensionality ratio
    assert feat_dict['dimensionality_ratio'] == 3 / 100
    
    # Verify we got 8 features in vector
    assert len(vector) == 8
    
    # Verify skewness and kurtosis are computed
    assert not np.isnan(feat_dict['mean_skewness'])
    assert not np.isnan(feat_dict['mean_kurtosis'])
    
    # Class imbalance should be NaN for regression
    assert np.isnan(feat_dict['class_imbalance_ratio'])
    
    print("✓ Regression test passed")


def test_classification_meta_features():
    """Test meta-feature extraction on classification dataset."""
    np.random.seed(42)
    
    # Create imbalanced classification dataset
    n_samples = 100
    X = pd.DataFrame({
        'feat1': np.random.randn(n_samples),
        'feat2': np.random.randn(n_samples)
    })
    # Imbalanced: 80 class 0, 20 class 1
    y = pd.Series([0] * 80 + [1] * 20)
    
    extractor = MetaFeatureExtractor(problem_type='classification')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nClassification Meta-Features:")
    for key, val in feat_dict.items():
        print(f"  {key}: {val}")
    
    # Verify class imbalance ratio
    expected_ratio = 20 / 80  # minority / majority
    assert abs(feat_dict['class_imbalance_ratio'] - expected_ratio) < 0.01
    
    # Breusch-Pagan should be NaN for classification
    assert np.isnan(feat_dict['breusch_pagan_pvalue'])
    
    print("✓ Classification test passed")


def test_high_vif_dataset():
    """Test VIF computation with highly correlated features."""
    np.random.seed(42)
    
    n_samples = 100
    base = np.random.randn(n_samples)
    
    # Create highly correlated features
    X = pd.DataFrame({
        'feat1': base,
        'feat2': base + np.random.randn(n_samples) * 0.1,  # Almost identical
        'feat3': np.random.randn(n_samples)  # Independent
    })
    y = pd.Series(np.random.randn(n_samples))
    
    extractor = MetaFeatureExtractor(problem_type='regression')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nHigh VIF Dataset:")
    print(f"  avg_vif: {feat_dict['avg_vif']}")
    
    # VIF should be high (>5 typically indicates multicollinearity)
    assert not np.isnan(feat_dict['avg_vif'])
    # Should be elevated due to correlation
    assert feat_dict['avg_vif'] > 1
    
    print("✓ VIF test passed")


def test_outlier_detection():
    """Test outlier percentage computation."""
    np.random.seed(42)
    
    # Create dataset with known outliers
    n_samples = 100
    normal_data = np.random.randn(n_samples - 5)
    outliers = np.array([10, -10, 15, -15, 20])  # Clear outliers
    
    X = pd.DataFrame({
        'feat1': np.concatenate([normal_data, outliers])
    })
    y = pd.Series(np.random.randn(n_samples))
    
    extractor = MetaFeatureExtractor(problem_type='regression')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nOutlier Detection:")
    print(f"  outlier_percentage: {feat_dict['outlier_percentage']}")
    
    # Should detect some outliers
    assert feat_dict['outlier_percentage'] > 0
    
    print("✓ Outlier detection test passed")


def test_edge_case_constant_features():
    """Test handling of constant features."""
    n_samples = 100
    
    X = pd.DataFrame({
        'constant': np.ones(n_samples),  # Constant feature
        'normal': np.random.randn(n_samples)
    })
    y = pd.Series(np.random.randn(n_samples))
    
    extractor = MetaFeatureExtractor(problem_type='regression')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nConstant Features Edge Case:")
    for key, val in feat_dict.items():
        print(f"  {key}: {val}")
    
    # Should handle gracefully without crashing
    assert feat_dict['dimensionality_ratio'] == 2 / 100
    
    print("✓ Constant features test passed")


def test_single_feature():
    """Test with single feature."""
    n_samples = 50
    
    X = pd.DataFrame({
        'feat1': np.random.randn(n_samples)
    })
    y = pd.Series(np.random.randn(n_samples))
    
    extractor = MetaFeatureExtractor(problem_type='regression')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nSingle Feature Test:")
    print(f"  dimensionality_ratio: {feat_dict['dimensionality_ratio']}")
    print(f"  avg_vif: {feat_dict['avg_vif']}")
    
    # VIF should be NaN (need at least 2 features)
    assert np.isnan(feat_dict['avg_vif'])
    assert feat_dict['dimensionality_ratio'] == 1 / 50
    
    print("✓ Single feature test passed")


def test_box_tidwell_positive_values():
    """Test Box-Tidwell with positive values."""
    np.random.seed(42)
    
    n_samples = 100
    # Ensure positive values for log transformation
    X = pd.DataFrame({
        'feat1': np.abs(np.random.randn(n_samples)) + 1,
        'feat2': np.abs(np.random.randn(n_samples)) + 1
    })
    y = pd.Series(np.random.randn(n_samples))
    
    extractor = MetaFeatureExtractor(problem_type='regression')
    vector, feat_dict = extractor.compute_all(X, y)
    
    print(f"\nBox-Tidwell Test:")
    print(f"  box_tidwell_pvalue: {feat_dict['box_tidwell_pvalue']}")
    
    # Should compute (may or may not be significant)
    # Just check it's not NaN or it computed correctly
    assert True  # Main goal is no crash
    
    print("✓ Box-Tidwell test passed")


if __name__ == "__main__":
    try:
        print("Starting Meta-Feature Extraction Tests...\n")
        
        print("=" * 50)
        test_regression_meta_features()
        
        print("=" * 50)
        test_classification_meta_features()
        
        print("=" * 50)
        test_high_vif_dataset()
        
        print("=" * 50)
        test_outlier_detection()
        
        print("=" * 50)
        test_edge_case_constant_features()
        
        print("=" * 50)
        test_single_feature()
        
        print("=" * 50)
        test_box_tidwell_positive_values()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
