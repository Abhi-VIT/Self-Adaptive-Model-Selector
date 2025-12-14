import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings


class MetaDatasetGenerator:
    """
    Generates synthetic meta-datasets for training the meta-learner.
    
    Simulates datasets with varying characteristics and determines
    the best-performing model based on heuristics.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the generator."""
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_meta_dataset(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic meta-dataset.
        
        Args:
            n_samples: Number of meta-samples to generate
            
        Returns:
            X_meta: Meta-features (n_samples, 8)
            y_meta: Best model labels (n_samples,)
        """
        X_meta = []
        y_meta = []
        
        for _ in range(n_samples):
            meta_features = self._generate_random_meta_features()
            best_model = self._simulate_best_model(meta_features)
            
            X_meta.append(meta_features)
            y_meta.append(best_model)
        
        return np.array(X_meta), np.array(y_meta)
    
    def _generate_random_meta_features(self) -> np.ndarray:
        """Generate random but realistic meta-features."""
        # Dimensionality ratio: 0.001 to 1.5
        dim_ratio = np.random.uniform(0.001, 1.5)
        
        # VIF: 1 to 50
        vif = np.random.uniform(1, 50)
        
        # P-values: 0 to 1
        bp_pvalue = np.random.uniform(0, 1)
        bt_pvalue = np.random.uniform(0, 1)
        
        # Skewness: -3 to 3
        skewness = np.random.uniform(-3, 3)
        
        # Kurtosis: -2 to 10
        kurtosis = np.random.uniform(-2, 10)
        
        # Outlier percentage: 0 to 30%
        outlier_pct = np.random.uniform(0, 30)
        
        # Class imbalance: 0.05 to 0.5 (or NaN for regression)
        if np.random.rand() > 0.5:  # 50% classification
            imbalance = np.random.uniform(0.05, 0.5)
        else:  # 50% regression
            imbalance = np.nan
        
        return np.array([
            dim_ratio, vif, bp_pvalue, bt_pvalue,
            skewness, kurtosis, outlier_pct, imbalance
        ])
    
    def _simulate_best_model(self, meta_features: np.ndarray) -> str:
        """
        Simulate which model would perform best based on heuristics.
        
        This uses domain knowledge to assign models based on meta-features.
        """
        dim_ratio, vif, bp_pvalue, bt_pvalue, skewness, kurtosis, outlier_pct, imbalance = meta_features
        
        is_classification = not np.isnan(imbalance)
        
        # Rule-based simulation of best model
        
        # High VIF + Heteroscedasticity → Tree-based
        if vif > 10 or bp_pvalue < 0.05:
            if is_classification:
                return "Random Forest Classifier"
            else:
                return "Random Forest Regressor"
        
        # Significant non-linearity → Tree or Neural
        if bt_pvalue < 0.05:
            if np.random.rand() > 0.5:
                if is_classification:
                    return "XGBoost Classifier"
                else:
                    return "XGBoost Regressor"
            else:
                if is_classification:
                    return "MLP Classifier"
                else:
                    return "MLP Regressor"
        
        # High dimensionality → SVM or Tree
        if dim_ratio > 0.5:
            if is_classification:
                return "Random Forest Classifier"
            else:
                return "Random Forest Regressor"
        
        # Low dimensionality + good data → Neural or Linear
        if dim_ratio < 0.05:
            if vif < 5 and bp_pvalue > 0.1:
                # Linear models work well
                if is_classification:
                    return "Logistic Regression"
                else:
                    return "Linear Regression"
            else:
                # Neural networks with enough data
                if is_classification:
                    return "MLP Classifier"
                else:
                    return "MLP Regressor"
        
        # Class imbalance → Tree-based
        if is_classification and imbalance < 0.2:
            return "XGBoost Classifier"
        
        # Default: use versatile models
        if is_classification:
            return "Random Forest Classifier"
        else:
            return "Random Forest Regressor"


class MetaLearner:
    """
    Meta-learner that predicts the best model for a given dataset
    based on its meta-features.
    """
    
    def __init__(self, meta_model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the meta-learner.
        
        Args:
            meta_model_type: 'random_forest' or 'gradient_boosting'
            random_state: Random seed
        """
        self.meta_model_type = meta_model_type
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='constant', fill_value=-999)  # Use constant for NaN
        self.is_trained = False
        
        if meta_model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif meta_model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
    
    def train(self, X_meta: np.ndarray, y_meta: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Train the meta-learner with cross-validation.
        
        Args:
            X_meta: Meta-features (n_samples, n_features)
            y_meta: Best model labels (n_samples,)
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        # Impute NaN values (e.g., class_imbalance_ratio for regression)
        X_meta_imputed = self.imputer.fit_transform(X_meta)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_meta)
        
        # Cross-validation
        cv_results = cross_validate(
            self.model, X_meta_imputed, y_encoded,
            cv=cv,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            return_train_score=True
        )
        
        # Train on full dataset
        self.model.fit(X_meta_imputed, y_encoded)
        self.is_trained = True
        
        # Compile metrics
        metrics = {
            'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
            'cv_accuracy_std': cv_results['test_accuracy'].std(),
            'cv_precision_mean': cv_results['test_precision_macro'].mean(),
            'cv_recall_mean': cv_results['test_recall_macro'].mean(),
            'cv_f1_mean': cv_results['test_f1_macro'].mean(),
            'train_accuracy_mean': cv_results['train_accuracy'].mean(),
        }
        
        return metrics
    
    def predict(self, meta_features: np.ndarray) -> str:
        """
        Predict the best model for given meta-features.
        
        Args:
            meta_features: Meta-feature vector (8,) or (n_samples, 8)
            
        Returns:
            Model name
        """
        if not self.is_trained:
            raise ValueError("Meta-learner not trained. Call train() first.")
        
        # Handle single sample
        if meta_features.ndim == 1:
            meta_features = meta_features.reshape(1, -1)
        
        # Impute NaN values
        meta_features_imputed = self.imputer.transform(meta_features)
        
        y_pred_encoded = self.model.predict(meta_features_imputed)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        if len(y_pred) == 1:
            return y_pred[0]
        return y_pred
    
    def predict_proba(self, meta_features: np.ndarray) -> Dict[str, float]:
        """
        Predict probability distribution over models.
        
        Args:
            meta_features: Meta-feature vector (8,)
            
        Returns:
            Dictionary mapping model names to probabilities
        """
        if not self.is_trained:
            raise ValueError("Meta-learner not trained. Call train() first.")
        
        if meta_features.ndim == 1:
            meta_features = meta_features.reshape(1, -1)
        
        # Impute NaN values
        meta_features_imputed = self.imputer.transform(meta_features)
        
        proba = self.model.predict_proba(meta_features_imputed)[0]
        
        # Map to model names
        proba_dict = {}
        for idx, prob in enumerate(proba):
            model_name = self.label_encoder.classes_[idx]
            proba_dict[model_name] = prob
        
        # Sort by probability
        proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
        
        return proba_dict
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for explainability.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Meta-learner not trained. Call train() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importances")
        
        feature_names = [
            'dimensionality_ratio',
            'avg_vif',
            'breusch_pagan_pvalue',
            'box_tidwell_pvalue',
            'mean_skewness',
            'mean_kurtosis',
            'outlier_percentage',
            'class_imbalance_ratio'
        ]
        
        importances = self.model.feature_importances_
        
        # Sort by importance
        importance_dict = dict(zip(feature_names, importances))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def save(self, filepath: str):
        """Save the trained meta-learner."""
        if not self.is_trained:
            warnings.warn("Saving untrained meta-learner")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'imputer': self.imputer,
                'meta_model_type': self.meta_model_type,
                'random_state': self.random_state,
                'is_trained': self.is_trained
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'MetaLearner':
        """Load a trained meta-learner."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        meta_learner = cls(
            meta_model_type=data['meta_model_type'],
            random_state=data['random_state']
        )
        meta_learner.model = data['model']
        meta_learner.label_encoder = data['label_encoder']
        meta_learner.imputer = data['imputer']
        meta_learner.is_trained = data['is_trained']
        
        return meta_learner
