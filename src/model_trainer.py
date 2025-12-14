import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import xgboost as xgb
import warnings


# Model Registry
MODEL_REGISTRY = {
    # Regression
    "Linear Regression": LinearRegression,
    "Random Forest Regressor": RandomForestRegressor,
    "XGBoost Regressor": xgb.XGBRegressor,
    "Support Vector Regressor (SVR)": SVR,
    "MLP Regressor": MLPRegressor,
    
    # Classification
    "Logistic Regression": LogisticRegression,
    "Random Forest Classifier": RandomForestClassifier,
    "XGBoost Classifier": xgb.XGBClassifier,
    "Support Vector Classifier (SVC)": SVC,
    "MLP Classifier": MLPClassifier,
}


# Hyperparameter Search Spaces
HYPERPARAMETER_SPACES = {
    "Random Forest Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
    },
    "Random Forest Classifier": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
    },
    "XGBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
    },
    "XGBoost Classifier": {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
    },
    "Support Vector Regressor (SVR)": {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
    },
    "Support Vector Classifier (SVC)": {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'probability': [True],
    },
    "MLP Regressor": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01],
    },
    "MLP Classifier": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001, 0.01],
    },
}


@dataclass
class TrainingResult:
    """Container for training results."""
    model_name: str
    model: Any
    problem_type: str
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    training_time: float
    test_size_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding model object)."""
        result = asdict(self)
        result.pop('model')  # Don't serialize model
        return result


class ModelTrainer:
    """
    Trains and evaluates ML models with hyperparameter tuning.
    """
    
    def __init__(self, model_name: str, problem_type: str, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            model_name: Name of the model to train
            problem_type: 'regression' or 'classification'
            random_state: Random seed
        """
        self.model_name = model_name
        self.problem_type = problem_type
        self.random_state = random_state
        
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        self.model_class = MODEL_REGISTRY[model_name]
    
    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        tune_hyperparameters: bool = True
    ) -> TrainingResult:
        """
        Train model with optional hyperparameter tuning and evaluate.
        
        Args:
            X: Features
            y: Target
            test_size: Fraction of data for testing
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            TrainingResult with model and metrics
        """
        start_time = time.time()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Train model
        if tune_hyperparameters and self.model_name in HYPERPARAMETER_SPACES:
            model, best_params = self._train_with_tuning(X_train, y_train)
        else:
            model = self._train_default(X_train, y_train)
            best_params = {}
        
        # Evaluate
        metrics = self._evaluate(model, X_test, y_test)
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            model_name=self.model_name,
            model=model,
            problem_type=self.problem_type,
            metrics=metrics,
            best_params=best_params,
            training_time=training_time,
            test_size_used=test_size
        )
    
    def _train_default(self, X_train, y_train):
        """Train model with default parameters."""
        if 'random_state' in self.model_class().get_params():
            model = self.model_class(random_state=self.random_state)
        else:
            model = self.model_class()
        
        model.fit(X_train, y_train)
        return model
    
    def _train_with_tuning(self, X_train, y_train) -> Tuple[Any, Dict]:
        """Train model with hyperparameter tuning."""
        param_space = HYPERPARAMETER_SPACES[self.model_name]
        
        # Base model
        if 'random_state' in self.model_class().get_params():
            base_model = self.model_class(random_state=self.random_state)
        else:
            base_model = self.model_class()
        
        # Scoring metric
        if self.problem_type == 'regression':
            scoring = 'neg_mean_squared_error'
        else:
            scoring = 'f1_weighted'
        
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_space,
            n_iter=min(10, len(list(param_space.values())[0])),  # Adaptive iterations
            cv=3,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        return search.best_estimator_, search.best_params_
    
    def _evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X_test)
        
        if self.problem_type == 'regression':
            return self._regression_metrics(y_test, y_pred)
        else:
            return self._classification_metrics(model, X_test, y_test, y_pred)
    
    def _regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Compute regression metrics."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handling zeros)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
    
    def _classification_metrics(self, model, X_test, y_test, y_pred) -> Dict[str, float]:
        """Compute classification metrics."""
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        # ROC-AUC (if probability available and binary)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    metrics['roc_auc'] = float(roc_auc)
        except:
            pass
        
        return metrics
