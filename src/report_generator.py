import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class ModelSelectionReport:
    """
    Comprehensive report for the model selection and training process.
    """
    # Dataset Information
    dataset_name: str
    problem_type: str
    n_samples: int
    n_features: int
    is_time_series: bool
    
    # Meta-features
    meta_features: Dict[str, float]
    
    # Model Selection
    selected_model: str
    selection_reasoning: Dict[str, Any]
    
    # Model Performance
    training_metrics: Dict[str, float]
    best_hyperparameters: Dict[str, Any]
    training_time: float
    
    # Explainability
    model_explainability: Dict[str, Any]
    system_explainability: Dict[str, Any]
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelSelectionReport':
        """Load report from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        summary = f"""
=== Model Selection Report ===
Dataset: {self.dataset_name}
Problem Type: {self.problem_type}
Samples: {self.n_samples}, Features: {self.n_features}

Selected Model: {self.selected_model}
Training Time: {self.training_time:.2f}s

Performance Metrics:
"""
        for metric, value in self.training_metrics.items():
            summary += f"  {metric}: {value:.4f}\n"
        
        summary += "\nKey Selection Factors:\n"
        for factor in self.selection_reasoning.get('key_meta_features', []):
            summary += f"  - {factor}\n"
        
        summary += "\nModel Strengths:\n"
        for strength in self.selection_reasoning.get('model_strengths', []):
            summary += f"  - {strength}\n"
        
        return summary


class ReportGenerator:
    """
    Generates comprehensive model selection reports.
    """
    
    @staticmethod
    def generate_report(
        dataset_name: str,
        dataset_info: Any,
        meta_features: Dict[str, float],
        selected_model: str,
        selection_reasoning: Dict[str, Any],
        training_result: Any,
        model_explainability: Dict[str, Any],
        system_explainability: Dict[str, Any]
    ) -> ModelSelectionReport:
        """
        Generate a complete model selection report.
        
        Args:
            dataset_name: Name of the dataset
            dataset_info: DatasetInfo object
            meta_features: Meta-feature dictionary
            selected_model: Name of selected model
            selection_reasoning: Why model was selected
            training_result: TrainingResult object
            model_explainability: Model-level explanations
            system_explainability: System-level explanations
            
        Returns:
            ModelSelectionReport
        """
        return ModelSelectionReport(
            dataset_name=dataset_name,
            problem_type=dataset_info.problem_type,
            n_samples=dataset_info.n_samples,
            n_features=dataset_info.n_features,
            is_time_series=dataset_info.is_time_series,
            meta_features=meta_features,
            selected_model=selected_model,
            selection_reasoning=selection_reasoning,
            training_metrics=training_result.metrics,
            best_hyperparameters=training_result.best_params,
            training_time=training_result.training_time,
            model_explainability=model_explainability,
            system_explainability=system_explainability
        )
