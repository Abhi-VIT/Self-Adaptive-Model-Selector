import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ModelType(Enum):
    """Model type categories."""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    SVM = "svm"
    NEURAL = "neural"
    SEQUENCE = "sequence"


@dataclass
class ModelCandidate:
    """Represents a candidate ML/DL model."""
    name: str
    model_type: ModelType
    problem_types: List[str]  # ['regression', 'classification']
    requires_time_series: bool = False
    
    def supports_problem(self, problem_type: str, is_time_series: bool = False) -> bool:
        """Check if model supports the problem type."""
        if self.requires_time_series and not is_time_series:
            return False
        return problem_type in self.problem_types


@dataclass
class FilterDecision:
    """Represents a filtering decision for a model."""
    model_name: str
    model_type: str
    included: bool
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'included': self.included,
            'reasons': self.reasons
        }


class ModelFilter:
    """
    Filters ML/DL models based on dataset characteristics and meta-features.
    
    Uses rule-based logic to intelligently select appropriate models.
    """
    
    # Thresholds for filtering rules
    HIGH_VIF_THRESHOLD = 10.0
    LOW_DIMENSIONALITY_THRESHOLD = 0.01
    HIGH_DIMENSIONALITY_THRESHOLD = 0.8
    NONLINEARITY_PVALUE_THRESHOLD = 0.05
    HETEROSCEDASTICITY_PVALUE_THRESHOLD = 0.05
    CLASS_IMBALANCE_THRESHOLD = 0.2
    
    def __init__(self):
        """Initialize the model filter."""
        self.model_pool = self._build_model_pool()
    
    def _build_model_pool(self) -> List[ModelCandidate]:
        """Define the complete pool of candidate models."""
        return [
            # Linear Models
            ModelCandidate(
                name="Linear Regression",
                model_type=ModelType.LINEAR,
                problem_types=["regression"]
            ),
            ModelCandidate(
                name="Logistic Regression",
                model_type=ModelType.LINEAR,
                problem_types=["classification"]
            ),
            
            # Tree-based Models
            ModelCandidate(
                name="Random Forest Regressor",
                model_type=ModelType.TREE_BASED,
                problem_types=["regression"]
            ),
            ModelCandidate(
                name="Random Forest Classifier",
                model_type=ModelType.TREE_BASED,
                problem_types=["classification"]
            ),
            ModelCandidate(
                name="XGBoost Regressor",
                model_type=ModelType.TREE_BASED,
                problem_types=["regression"]
            ),
            ModelCandidate(
                name="XGBoost Classifier",
                model_type=ModelType.TREE_BASED,
                problem_types=["classification"]
            ),
            
            # SVM Models
            ModelCandidate(
                name="Support Vector Regressor (SVR)",
                model_type=ModelType.SVM,
                problem_types=["regression"]
            ),
            ModelCandidate(
                name="Support Vector Classifier (SVC)",
                model_type=ModelType.SVM,
                problem_types=["classification"]
            ),
            
            # Neural Network Models
            ModelCandidate(
                name="MLP Regressor",
                model_type=ModelType.NEURAL,
                problem_types=["regression"]
            ),
            ModelCandidate(
                name="MLP Classifier",
                model_type=ModelType.NEURAL,
                problem_types=["classification"]
            ),
            
            # Sequence Models (Time-series)
            ModelCandidate(
                name="LSTM",
                model_type=ModelType.SEQUENCE,
                problem_types=["regression", "classification"],
                requires_time_series=True
            ),
            ModelCandidate(
                name="GRU",
                model_type=ModelType.SEQUENCE,
                problem_types=["regression", "classification"],
                requires_time_series=True
            ),
        ]
    
    def filter_models(
        self,
        meta_features: Dict[str, float],
        dataset_info: Any
    ) -> List[FilterDecision]:
        """
        Filter models based on meta-features and dataset info.
        
        Args:
            meta_features: Dictionary of meta-features from MetaFeatureExtractor
            dataset_info: DatasetInfo object from DataProcessor
            
        Returns:
            List of FilterDecision objects with inclusion status and reasons
        """
        decisions = []
        
        problem_type = dataset_info.problem_type
        is_time_series = dataset_info.is_time_series
        
        for model in self.model_pool:
            # Check if model supports this problem type
            if not model.supports_problem(problem_type, is_time_series):
                continue
            
            decision = self._apply_rules(model, meta_features, dataset_info)
            decisions.append(decision)
        
        return decisions
    
    def _apply_rules(
        self,
        model: ModelCandidate,
        meta_features: Dict[str, float],
        dataset_info: Any
    ) -> FilterDecision:
        """
        Apply filtering rules to determine if model should be included.
        
        Returns a FilterDecision with reasons for inclusion/exclusion.
        """
        reasons = []
        exclude_reasons = []
        
        # Extract meta-features
        vif = meta_features.get('avg_vif', np.nan)
        dim_ratio = meta_features.get('dimensionality_ratio', np.nan)
        box_tidwell_p = meta_features.get('box_tidwell_pvalue', np.nan)
        bp_pvalue = meta_features.get('breusch_pagan_pvalue', np.nan)
        imbalance_ratio = meta_features.get('class_imbalance_ratio', np.nan)
        
        # Rule 1: High VIF → Exclude linear models
        if not np.isnan(vif) and vif > self.HIGH_VIF_THRESHOLD:
            if model.model_type == ModelType.LINEAR:
                exclude_reasons.append(
                    f"High multicollinearity detected (VIF={vif:.2f}), linear models may be unstable"
                )
            elif model.model_type == ModelType.TREE_BASED:
                reasons.append("Tree-based models handle multicollinearity well")
        
        # Rule 2: Non-linearity detected → Prefer tree-based/neural, exclude linear
        if not np.isnan(box_tidwell_p) and box_tidwell_p < self.NONLINEARITY_PVALUE_THRESHOLD:
            if model.model_type == ModelType.LINEAR:
                exclude_reasons.append(
                    f"Significant non-linearity detected (p={box_tidwell_p:.4f}), linear assumptions violated"
                )
            elif model.model_type in [ModelType.TREE_BASED, ModelType.NEURAL]:
                reasons.append("Non-linear relationships detected, model handles this well")
        
        # Rule 3: High dimensionality with small samples → Exclude deep models
        if not np.isnan(dim_ratio) and dim_ratio > self.HIGH_DIMENSIONALITY_THRESHOLD:
            if model.model_type == ModelType.NEURAL:
                exclude_reasons.append(
                    f"High dimensionality ratio ({dim_ratio:.2f}), neural networks may overfit on small datasets"
                )
            elif model.model_type == ModelType.TREE_BASED:
                reasons.append("Tree-based models handle high-dimensional data effectively")
        
        # Rule 4: Low dimensionality → Favor neural networks
        if not np.isnan(dim_ratio) and dim_ratio < self.LOW_DIMENSIONALITY_THRESHOLD:
            if model.model_type == ModelType.NEURAL:
                reasons.append("Large dataset relative to features, neural networks can leverage data volume")
        
        # Rule 5: Heteroscedasticity → Exclude linear regression
        if (not np.isnan(bp_pvalue) and 
            bp_pvalue < self.HETEROSCEDASTICITY_PVALUE_THRESHOLD and
            model.name == "Linear Regression"):
            exclude_reasons.append(
                f"Heteroscedasticity detected (BP p={bp_pvalue:.4f}), violates linear regression assumptions"
            )
        
        # Rule 6: Class imbalance → Prefer tree-based models
        if (not np.isnan(imbalance_ratio) and 
            imbalance_ratio < self.CLASS_IMBALANCE_THRESHOLD and
            dataset_info.problem_type == "classification"):
            if model.model_type == ModelType.TREE_BASED:
                reasons.append(
                    f"High class imbalance (ratio={imbalance_ratio:.2f}), tree models handle this better"
                )
        
        # Rule 7: Time-series → Include sequence models
        if dataset_info.is_time_series:
            if model.model_type == ModelType.SEQUENCE:
                reasons.append("Time-series data detected, sequence models are ideal")
        
        # Default positive reasons if no specific ones
        if len(reasons) == 0 and len(exclude_reasons) == 0:
            reasons.append("General-purpose model suitable for this problem type")
        
        # Determine inclusion
        included = len(exclude_reasons) == 0
        final_reasons = exclude_reasons if not included else reasons
        
        return FilterDecision(
            model_name=model.name,
            model_type=model.model_type.value,
            included=included,
            reasons=final_reasons
        )
    
    def get_included_models(
        self,
        meta_features: Dict[str, float],
        dataset_info: Any
    ) -> List[str]:
        """
        Get list of included model names only.
        
        Returns:
            List of model names that passed filtering
        """
        decisions = self.filter_models(meta_features, dataset_info)
        return [d.model_name for d in decisions if d.included]
    
    def get_summary(
        self,
        meta_features: Dict[str, float],
        dataset_info: Any
    ) -> Dict[str, Any]:
        """
        Get a summary of filtering results.
        
        Returns:
            Dictionary with included/excluded models and statistics
        """
        decisions = self.filter_models(meta_features, dataset_info)
        
        included = [d for d in decisions if d.included]
        excluded = [d for d in decisions if not d.included]
        
        return {
            'total_candidates': len(decisions),
            'included_count': len(included),
            'excluded_count': len(excluded),
            'included_models': [d.to_dict() for d in included],
            'excluded_models': [d.to_dict() for d in excluded]
        }
