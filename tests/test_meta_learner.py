import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_learner import MetaDatasetGenerator, MetaLearner


def test_meta_dataset_generation():
    """Test synthetic meta-dataset generation."""
    print("\n" + "="*60)
    print("Test: Meta-dataset generation")
    print("="*60)
    
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=100)
    
    print(f"\nGenerated meta-dataset:")
    print(f"  X_meta shape: {X_meta.shape}")
    print(f"  y_meta shape: {y_meta.shape}")
    print(f"  Unique models: {len(np.unique(y_meta))}")
    print(f"  Model distribution:")
    unique, counts = np.unique(y_meta, return_counts=True)
    for model, count in zip(unique, counts):
        print(f"    {model}: {count}")
    
    # Verify shapes
    assert X_meta.shape == (100, 8), "X_meta should have 8 features"
    assert y_meta.shape == (100,), "y_meta should have 100 labels"
    assert len(np.unique(y_meta)) > 1, "Should have multiple model types"
    
    print("\n✓ Meta-dataset generation test passed")


def test_meta_learner_training():
    """Test meta-learner training with cross-validation."""
    print("\n" + "="*60)
    print("Test: Meta-learner training")
    print("="*60)
    
    # Generate training data
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=200)
    
    # Train meta-learner
    meta_learner = MetaLearner(meta_model_type='random_forest', random_state=42)
    metrics = meta_learner.train(X_meta, y_meta, cv=5)
    
    print(f"\nTraining metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # Verify reasonable performance
    assert metrics['cv_accuracy_mean'] > 0.3, "Should have > 30% accuracy"
    assert meta_learner.is_trained, "Should be marked as trained"
    
    print("\n✓ Meta-learner training test passed")


def test_meta_learner_prediction():
    """Test prediction on new meta-features."""
    print("\n" + "="*60)
    print("Test: Meta-learner prediction")
    print("="*60)
    
    # Generate and train
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=200)
    
    meta_learner = MetaLearner(meta_model_type='random_forest', random_state=42)
    meta_learner.train(X_meta, y_meta, cv=3)
    
    # Create test meta-features (high VIF scenario)
    test_features = np.array([
        0.05,  # low dim ratio
        15.0,  # high VIF
        0.5,   # BP p-value
        0.3,   # BT p-value
        0.1,   # skewness
        0.2,   # kurtosis
        5.0,   # outlier %
        np.nan # regression
    ])
    
    # Predict
    prediction = meta_learner.predict(test_features)
    print(f"\nPrediction for high-VIF regression dataset:")
    print(f"  Best model: {prediction}")
    
    # Get probabilities
    proba = meta_learner.predict_proba(test_features)
    print(f"\nTop 3 model probabilities:")
    for i, (model, prob) in enumerate(list(proba.items())[:3]):
        print(f"  {i+1}. {model}: {prob:.4f}")
    
    assert isinstance(prediction, str), "Should return model name"
    assert sum(proba.values()) > 0.99, "Probabilities should sum to ~1"
    
    print("\n✓ Meta-learner prediction test passed")


def test_feature_importance():
    """Test feature importance extraction."""
    print("\n" + "="*60)
    print("Test: Feature importance")
    print("="*60)
    
    # Generate and train
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=200)
    
    meta_learner = MetaLearner(meta_model_type='random_forest', random_state=42)
    meta_learner.train(X_meta, y_meta, cv=3)
    
    # Get importance
    importance = meta_learner.get_feature_importance()
    
    print(f"\nFeature importance (top 5):")
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    assert len(importance) == 8, "Should have 8 feature importances"
    assert all(v >= 0 for v in importance.values()), "Importances should be non-negative"
    
    print("\n✓ Feature importance test passed")


def test_save_load():
    """Test model persistence."""
    print("\n" + "="*60)
    print("Test: Save and load meta-learner")
    print("="*60)
    
    # Generate and train
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=100)
    
    meta_learner = MetaLearner(meta_model_type='random_forest', random_state=42)
    meta_learner.train(X_meta, y_meta, cv=3)
    
    # Test prediction before save
    test_features = np.array([0.05, 15.0, 0.5, 0.3, 0.1, 0.2, 5.0, np.nan])
    pred_before = meta_learner.predict(test_features)
    
    # Save
    model_path = os.path.join('..', 'models', 'test_meta_learner.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    meta_learner.save(model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Load
    loaded_learner = MetaLearner.load(model_path)
    pred_after = loaded_learner.predict(test_features)
    
    print(f"\nPrediction before save: {pred_before}")
    print(f"Prediction after load: {pred_after}")
    
    assert pred_before == pred_after, "Predictions should match after save/load"
    assert loaded_learner.is_trained, "Loaded model should be marked as trained"
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n✓ Save/load test passed")


def test_gradient_boosting_meta_learner():
    """Test with gradient boosting meta-learner."""
    print("\n" + "="*60)
    print("Test: Gradient Boosting meta-learner")
    print("="*60)
    
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=150)
    
    meta_learner = MetaLearner(meta_model_type='gradient_boosting', random_state=42)
    metrics = meta_learner.train(X_meta, y_meta, cv=3)
    
    print(f"\nGradient Boosting metrics:")
    print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.4f}")
    print(f"  CV F1: {metrics['cv_f1_mean']:.4f}")
    
    assert metrics['cv_accuracy_mean'] > 0.25, "Should have reasonable accuracy"
    
    print("\n✓ Gradient Boosting test passed")


def test_batch_prediction():
    """Test prediction on multiple samples at once."""
    print("\n" + "="*60)
    print("Test: Batch prediction")
    print("="*60)
    
    generator = MetaDatasetGenerator(random_state=42)
    X_meta, y_meta = generator.generate_meta_dataset(n_samples=200)
    
    meta_learner = MetaLearner(meta_model_type='random_forest', random_state=42)
    meta_learner.train(X_meta, y_meta, cv=3)
    
    # Create batch of test features
    test_batch = np.array([
        [0.05, 15.0, 0.5, 0.3, 0.1, 0.2, 5.0, np.nan],  # High VIF regression
        [0.8, 2.0, 0.5, 0.3, 0.1, 0.2, 5.0, 0.1],       # High dim classification
        [0.02, 2.0, 0.5, 0.01, 0.1, 0.2, 5.0, np.nan],  # Low dim + non-linear
    ])
    
    predictions = meta_learner.predict(test_batch)
    
    print(f"\nBatch predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {pred}")
    
    assert len(predictions) == 3, "Should predict for all samples"
    
    print("\n✓ Batch prediction test passed")


if __name__ == "__main__":
    try:
        print("\n" + "="*60)
        print("Starting Meta-Learner Tests")
        print("="*60)
        
        test_meta_dataset_generation()
        test_meta_learner_training()
        test_meta_learner_prediction()
        test_feature_importance()
        test_save_load()
        test_gradient_boosting_meta_learner()
        test_batch_prediction()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
