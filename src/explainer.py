import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional
import warnings


class ModelExplainer:
    """
    Provides model-level explanations using SHAP.
    """
    
    def __init__(self, model, X: pd.DataFrame, problem_type: str = 'regression'):
        """
        Initialize explainer.
        
        Args:
            model: Trained model
            X: Training/reference data
            problem_type: 'regression' or 'classification'
        """
        self.model = model
        self.X = X
        self.problem_type = problem_type
        self.explainer = None
        self.shap_values = None
        
    def explain_with_shap(self, X_explain: Optional[pd.DataFrame] = None, max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Args:
            X_explain: Data to explain (if None, uses training data sample)
            max_samples: Maximum samples to explain (for performance)
            
        Returns:
            Dictionary with SHAP values and feature importance
        """
        if X_explain is None:
            # Use sample of training data
            X_explain = self.X.sample(min(max_samples, len(self.X)), random_state=42)
        else:
            X_explain = X_explain.sample(min(max_samples, len(X_explain)), random_state=42)
        
        try:
            # Try TreeExplainer first (works for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_explain)
        except:
            try:
                # Try LinearExplainer (works for linear models)
                self.explainer = shap.LinearExplainer(self.model, self.X)
                self.shap_values = self.explainer.shap_values(X_explain)
            except:
                # Fallback to KernelExplainer (slower but universal)
                warnings.warn("Using KernelExplainer (slower). Consider using tree-based or linear models.")
                sample_background = shap.sample(self.X, min(50, len(self.X)))
                self.explainer = shap.KernelExplainer(self.model.predict, sample_background)
                self.shap_values = self.explainer.shap_values(X_explain)
        
        # Compute feature importance from SHAP values
        feature_importance = self._compute_shap_importance(X_explain)
        
        return {
            'shap_values_shape': np.array(self.shap_values).shape if isinstance(self.shap_values, np.ndarray) else 'computed',
            'feature_importance': feature_importance,
            'num_samples_explained': len(X_explain),
            'explainer_type': type(self.explainer).__name__
        }
   
    def _compute_shap_importance(self, X_explain: pd.DataFrame) -> Dict[str, float]:
        """Compute feature importance from SHAP values."""
        # Handle multi-output SHAP values (e.g., for multiclass)
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(self.shap_values[0])  # Use first class
        else:
            shap_vals = np.abs(self.shap_values)
        
        # Mean absolute SHAP value for each feature
        importance = np.mean(shap_vals, axis=0)
        
        # Create dictionary
        feature_names = X_explain.columns if hasattr(X_explain, 'columns') else [f'feature_{i}' for i in range(len(importance))]
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from model (if available)."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.X.columns if hasattr(self.X, 'columns') else [f'feature_{i}' for i in range(len(importances))]
            importance_dict = dict(zip(feature_names, importances))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coefs = np.abs(self.model.coef_)
            if coefs.ndim > 1:
                coefs = np.mean(np.abs(coefs), axis=0)
            feature_names = self.X.columns if hasattr(self.X, 'columns') else [f'feature_{i}' for i in range(len(coefs))]
            importance_dict = dict(zip(feature_names, coefs))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}


class SystemExplainer:
    """
    Provides system-level explanations for model selection.
    """
    
    @staticmethod
    def explain_selection(
        selected_model: str,
        meta_features: Dict[str, float],
        meta_learner_importance: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Explain why a particular model was selected.
        
        Args:
            selected_model: Name of selected model
            meta_features: Dictionary of meta-features
            meta_learner_importance: Feature importance from meta-learner
            
        Returns:
            Dictionary with selection reasoning
        """
        reasons = []
        key_factors = []
        
        # Analyze meta-features
        vif = meta_features.get('avg_vif', 0)
        dim_ratio = meta_features.get('dimensionality_ratio', 0)
        bt_pvalue = meta_features.get('box_tidwell_pvalue', 1)
        bp_pvalue = meta_features.get('breusch_pagan_pvalue', 1)
        imbalance = meta_features.get('class_imbalance_ratio', 0.5)
        
        # Generate reasons based on meta-features
        if vif > 10:
            reasons.append(f"High multicollinearity detected (VIF={vif:.2f})")
            key_factors.append('avg_vif')
        
        if bt_pvalue < 0.05:
            reasons.append(f"Significant non-linearity detected (p={bt_pvalue:.4f})")
            key_factors.append('box_tidwell_pvalue')
        
        if dim_ratio > 0.5:
            reasons.append(f"High dimensionality ratio ({dim_ratio:.2f})")
            key_factors.append('dimensionality_ratio')
        elif dim_ratio < 0.05:
            reasons.append(f"Large dataset relative to features (ratio={dim_ratio:.3f})")
            key_factors.append('dimensionality_ratio')
        
        if bp_pvalue < 0.05:
            reasons.append(f"Heteroscedasticity detected (BP p={bp_pvalue:.4f})")
            key_factors.append('breusch_pagan_pvalue')
        
        if not np.isnan(imbalance) and imbalance < 0.2:
            reasons.append(f"Class imbalance detected (ratio={imbalance:.2f})")
            key_factors.append('class_imbalance_ratio')
        
        # Model-specific reasoning
        model_characteristics = SystemExplainer._get_model_characteristics(selected_model)
        
        return {
            'selected_model': selected_model,
            'data_characteristics': reasons,
            'key_meta_features': key_factors,
            'model_strengths': model_characteristics,
            'meta_learner_top_features': list(meta_learner_importance.keys())[:3] if meta_learner_importance else []
        }
    
    @staticmethod
    def _get_model_characteristics(model_name: str) -> List[str]:
        """Get key characteristics of the model."""
        characteristics = {
            "Random Forest Regressor": [
                "Handles non-linearity well",
                "Robust to multicollinearity",
                "Built-in feature importance"
            ],
            "Random Forest Classifier": [
                "Handles imbalanced classes well",
                "Robust to outliers",
                "No feature scaling required"
            ],
            "XGBoost Regressor": [
                "State-of-the-art gradient boosting",
                "Handles missing values natively",
                "Regularization to prevent overfitting"
            ],
            "XGBoost Classifier": [
                "Excellent for imbalanced datasets",
                "Fast training with parallelization",
                "Built-in cross-validation"
            ],
            "Linear Regression": [
                "Simple and interpretable",
                "Fast training and prediction",
                "Works well with linear relationships"
            ],
            "Logistic Regression": [
                "Probabilistic predictions",
                "Well-calibrated probabilities",
                "Regularization available"
            ],
            "MLP Regressor": [
                "Learns complex non-linear patterns",
                "Flexible architecture",
                "Works well with large datasets"
            ],
            "MLP Classifier": [
                "Deep learning capabilities",
                "Handles high-dimensional data",
                "Non-linear decision boundaries"
            ],
        }
        
        return characteristics.get(model_name, ["General-purpose machine learning model"])
