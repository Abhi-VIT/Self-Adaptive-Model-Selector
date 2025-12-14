import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_selector import ModelFilter, FilterDecision
from data_processor import DatasetInfo


def test_high_vif_excludes_linear():
    """Test that high VIF excludes linear models."""
    print("\n" + "="*60)
    print("Test: High VIF should exclude linear models")
    print("="*60)
    
    # Create meta-features with high VIF
    meta_features = {
        'dimensionality_ratio': 0.05,
        'avg_vif': 15.0,  # High VIF
        'breusch_pagan_pvalue': 0.5,
        'box_tidwell_pvalue': 0.3,
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': np.nan
    }
    
    dataset_info = DatasetInfo(
        problem_type='regression',
        n_samples=100,
        n_features=5,
        is_time_series=False
    )
    
    filter = ModelFilter()
    decisions = filter.filter_models(meta_features, dataset_info)
    
    # Check Linear Regression is excluded
    linear_decision = [d for d in decisions if d.model_name == "Linear Regression"][0]
    print(f"\nLinear Regression:")
    print(f"  Included: {linear_decision.included}")
    print(f"  Reasons: {linear_decision.reasons}")
    
    assert not linear_decision.included, "Linear Regression should be excluded with high VIF"
    assert any("multicollinearity" in r.lower() for r in linear_decision.reasons)
    
    # Check tree-based models are included
    rf_decision = [d for d in decisions if d.model_name == "Random Forest Regressor"][0]
    print(f"\nRandom Forest Regressor:")
    print(f"  Included: {rf_decision.included}")
    print(f"  Reasons: {rf_decision.reasons}")
    
    assert rf_decision.included, "Random Forest should be included"
    
    print("\n✓ High VIF test passed")


def test_nonlinearity_prefers_tree_models():
    """Test that non-linearity detection prefers tree-based models."""
    print("\n" + "="*60)
    print("Test: Non-linearity should exclude linear, prefer tree-based")
    print("="*60)
    
    meta_features = {
        'dimensionality_ratio': 0.05,
        'avg_vif': 2.0,
        'breusch_pagan_pvalue': 0.5,
        'box_tidwell_pvalue': 0.01,  # Significant non-linearity
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': np.nan
    }
    
    dataset_info = DatasetInfo(
        problem_type='regression',
        n_samples=100,
        n_features=5,
        is_time_series=False
    )
    
    filter = ModelFilter()
    decisions = filter.filter_models(meta_features, dataset_info)
    
    linear_decision = [d for d in decisions if d.model_name == "Linear Regression"][0]
    xgb_decision = [d for d in decisions if d.model_name == "XGBoost Regressor"][0]
    
    print(f"\nLinear Regression: {linear_decision.included}")
    print(f"  Reasons: {linear_decision.reasons}")
    print(f"\nXGBoost Regressor: {xgb_decision.included}")
    print(f"  Reasons: {xgb_decision.reasons}")
    
    assert not linear_decision.included, "Linear models should be excluded with non-linearity"
    assert xgb_decision.included, "Tree models should be included"
    assert any("non-linear" in r.lower() for r in xgb_decision.reasons)
    
    print("\n✓ Non-linearity test passed")


def test_time_series_includes_sequence_models():
    """Test that time-series detection includes LSTM/GRU."""
    print("\n" + "="*60)
    print("Test: Time-series should include LSTM/GRU")
    print("="*60)
    
    meta_features = {
        'dimensionality_ratio': 0.05,
        'avg_vif': 2.0,
        'breusch_pagan_pvalue': 0.5,
        'box_tidwell_pvalue': 0.3,
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': np.nan
    }
    
    dataset_info = DatasetInfo(
        problem_type='regression',
        n_samples=100,
        n_features=5,
        is_time_series=True  # Time-series
    )
    
    filter = ModelFilter()
    decisions = filter.filter_models(meta_features, dataset_info)
    
    # Check LSTM and GRU are in decisions
    lstm_decision = [d for d in decisions if d.model_name == "LSTM"][0]
    gru_decision = [d for d in decisions if d.model_name == "GRU"][0]
    
    print(f"\nLSTM:")
    print(f"  Included: {lstm_decision.included}")
    print(f"  Reasons: {lstm_decision.reasons}")
    
    print(f"\nGRU:")
    print(f"  Included: {gru_decision.included}")
    print(f"  Reasons: {gru_decision.reasons}")
    
    assert lstm_decision.included, "LSTM should be included for time-series"
    assert gru_decision.included, "GRU should be included for time-series"
    assert any("time-series" in r.lower() for r in lstm_decision.reasons)
    
    print("\n✓ Time-series test passed")


def test_high_dimensionality_excludes_neural():
    """Test that high dimensionality excludes neural networks."""
    print("\n" + "="*60)
    print("Test: High dimensionality should exclude neural networks")
    print("="*60)
    
    meta_features = {
        'dimensionality_ratio': 0.9,  # Very high
        'avg_vif': 2.0,
        'breusch_pagan_pvalue': 0.5,
        'box_tidwell_pvalue': 0.3,
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': np.nan
    }
    
    dataset_info = DatasetInfo(
        problem_type='regression',
        n_samples=100,
        n_features=90,
        is_time_series=False
    )
    
    filter = ModelFilter()
    decisions = filter.filter_models(meta_features, dataset_info)
    
    mlp_decision = [d for d in decisions if d.model_name == "MLP Regressor"][0]
    rf_decision = [d for d in decisions if d.model_name == "Random Forest Regressor"][0]
    
    print(f"\nMLP Regressor:")
    print(f"  Included: {mlp_decision.included}")
    print(f"  Reasons: {mlp_decision.reasons}")
    
    print(f"\nRandom Forest Regressor:")
    print(f"  Included: {rf_decision.included}")
    print(f"  Reasons: {rf_decision.reasons}")
    
    assert not mlp_decision.included, "MLP should be excluded with high dimensionality"
    assert any("dimensionality" in r.lower() for r in mlp_decision.reasons)
    assert rf_decision.included, "Random Forest should still be included"
    
    print("\n✓ High dimensionality test passed")


def test_class_imbalance_prefers_trees():
    """Test that class imbalance prefers tree-based models."""
    print("\n" + "="*60)
    print("Test: Class imbalance should prefer tree-based models")
    print("="*60)
    
    meta_features = {
        'dimensionality_ratio': 0.05,
        'avg_vif': 2.0,
        'breusch_pagan_pvalue': np.nan,
        'box_tidwell_pvalue': np.nan,
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': 0.1  # High imbalance (10:1)
    }
    
    dataset_info = DatasetInfo(
        problem_type='classification',
        classification_type='binary',
        n_samples=100,
        n_features=5,
        is_time_series=False
    )
    
    filter = ModelFilter()
    decisions = filter.filter_models(meta_features, dataset_info)
    
    rf_decision = [d for d in decisions if d.model_name == "Random Forest Classifier"][0]
    
    print(f"\nRandom Forest Classifier:")
    print(f"  Included: {rf_decision.included}")
    print(f"  Reasons: {rf_decision.reasons}")
    
    assert rf_decision.included
    assert any("imbalance" in r.lower() for r in rf_decision.reasons)
    
    print("\n✓ Class imbalance test passed")


def test_heteroscedasticity_excludes_linear_regression():
    """Test that heteroscedasticity excludes linear regression."""
    print("\n" + "="*60)
    print("Test: Heteroscedasticity should exclude Linear Regression")
    print("="*60)
    
    meta_features = {
        'dimensionality_ratio': 0.05,
        'avg_vif': 2.0,
        'breusch_pagan_pvalue': 0.01,  # Significant heteroscedasticity
        'box_tidwell_pvalue': 0.3,
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': np.nan
    }
    
    dataset_info = DatasetInfo(
        problem_type='regression',
        n_samples=100,
        n_features=5,
        is_time_series=False
    )
    
    filter = ModelFilter()
    decisions = filter.filter_models(meta_features, dataset_info)
    
    linear_decision = [d for d in decisions if d.model_name == "Linear Regression"][0]
    
    print(f"\nLinear Regression:")
    print(f"  Included: {linear_decision.included}")
    print(f"  Reasons: {linear_decision.reasons}")
    
    assert not linear_decision.included
    assert any("heteroscedasticity" in r.lower() for r in linear_decision.reasons)
    
    print("\n✓ Heteroscedasticity test passed")


def test_get_summary():
    """Test summary generation."""
    print("\n" + "="*60)
    print("Test: Summary generation")
    print("="*60)
    
    meta_features = {
        'dimensionality_ratio': 0.05,
        'avg_vif': 15.0,  # Will exclude linear
        'breusch_pagan_pvalue': 0.5,
        'box_tidwell_pvalue': 0.3,
        'mean_skewness': 0.1,
        'mean_kurtosis': 0.2,
        'outlier_percentage': 5.0,
        'class_imbalance_ratio': np.nan
    }
    
    dataset_info = DatasetInfo(
        problem_type='regression',
        n_samples=100,
        n_features=5,
        is_time_series=False
    )
    
    filter = ModelFilter()
    summary = filter.get_summary(meta_features, dataset_info)
    
    print(f"\nSummary:")
    print(f"  Total candidates: {summary['total_candidates']}")
    print(f"  Included: {summary['included_count']}")
    print(f"  Excluded: {summary['excluded_count']}")
    print(f"\nIncluded models:")
    for model in summary['included_models']:
        print(f"  - {model['model_name']}")
    
    assert summary['total_candidates'] > 0
    assert summary['included_count'] + summary['excluded_count'] == summary['total_candidates']
    
    print("\n✓ Summary test passed")


if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("Starting Model Filtering Tests")
        print("="*60)
        
        test_high_vif_excludes_linear()
        test_nonlinearity_prefers_tree_models()
        test_time_series_includes_sequence_models()
        test_high_dimensionality_excludes_neural()
        test_class_imbalance_prefers_trees()
        test_heteroscedasticity_excludes_linear_regression()
        test_get_summary()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
