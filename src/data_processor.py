import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Union, Dict, Tuple, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

@dataclass
class DatasetInfo:
    """Structured container for dataset metadata and problem type."""
    problem_type: str = "unknown"  # regression, classification
    classification_type: Optional[str] = None  # binary, multiclass, none
    n_samples: int = 0
    n_features: int = 0
    missing_percentage: float = 0.0
    is_time_series: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataProcessor:
    """
    Handles data loading, problem type detection, metadata extraction,
    and minimal preprocessing for the Self-Adaptive Model Selector.
    """
    
    def __init__(self, target_col: str):
        """
        Initialize the DataProcessor.
        
        Args:
            target_col: Name of the target variable column.
        """
        self.target_col = target_col
        self.label_encoder = None
        self.num_imputer = None
        self.cat_imputer = None
        self.feature_encoders = {}  # Dictionary to store label encoders for each categorical feature
        
    def load_data(self, source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load dataset from a CSV path or existing DataFrame.
        
        Args:
            source: File path (str) or pandas DataFrame.
            
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        if isinstance(source, str):
            try:
                df = pd.read_csv(source)
                # Try to parse potential date columns for time series detection later
                # We do a quick check for 'date' or 'time' in column names
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except (ValueError, TypeError):
                            pass
                return df
            except Exception as e:
                raise IOError(f"Error loading CSV from {source}: {e}")
        elif isinstance(source, pd.DataFrame):
            return source.copy()
        else:
            raise ValueError("Source must be a file path string or pandas DataFrame.")

    def _detect_problem_type(self, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
        """
        Detect if the problem is regression or classification.
        Also distinguishes between binary and multiclass classification.
        
        Args:
            df: The dataset.
            
        Returns:
            Tuple[str, Optional[str]]: (problem_type, classification_type)
        """
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")
            
        target = df[self.target_col]
        unique_values = target.nunique()
        total_samples = len(target)
        dtype = target.dtype
        
        # Simple heuristic for classification:
        # 1. If dtype is object/string/bool/categorical -> Classification
        # 2. If numeric but very few unique values (< 20 and < 10% of samples) -> Classification
        
        is_classification = False
        
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            is_classification = True
        elif pd.api.types.is_numeric_dtype(dtype):
            # Heuristic: If few unique integers, treat as classification
            if unique_values < 20 and (unique_values / total_samples) < 0.1:
                # Check if values are essentially integers
                if np.all(np.mod(target.dropna(), 1) == 0):
                    is_classification = True
        
        if is_classification:
            problem_type = "classification"
            if unique_values == 2:
                clf_type = "binary"
            else:
                clf_type = "multiclass"
        else:
            problem_type = "regression"
            clf_type = None
            
        return problem_type, clf_type

    def _detect_time_series(self, df: pd.DataFrame) -> bool:
        """
        Detect if the dataset has time-series characteristics.
        
        Args:
            df: The dataset.
            
        Returns:
            bool: True if time series detected.
        """
        # Check for DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            return True
            
        # Check for any datetime column that is sorted
        for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
            if df[col].is_monotonic_increasing:
                return True
                
        return False

    def _compute_missing_percentage(self, df: pd.DataFrame) -> float:
        """Calculate total percentage of missing values in the dataframe."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        return (missing_cells / total_cells) * 100 if total_cells > 0 else 0.0

    def process_and_analyze(self, source: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, DatasetInfo]:
        """
        Main pipeline: Load -> Detect/Analyze -> Preprocess.
        
        Args:
            source: Data source.
            
        Returns:
            X (pd.DataFrame): Processed features.
            y (pd.Series): Processed target.
            info (DatasetInfo): Extracted metadata.
        """
        df = self.load_data(source)
        
        # 1. Metadata & Detection
        problem_type, clf_type = self._detect_problem_type(df)
        is_ts = self._detect_time_series(df)
        missing_pct = self._compute_missing_percentage(df)
        
        # 2. Separation
        y = df[self.target_col].copy()
        X = df.drop(columns=[self.target_col]).copy()
        
        # Remove datetime columns as they cannot be used by ML models
        datetime_cols = X.select_dtypes(include=['datetime', 'datetimetz']).columns
        if len(datetime_cols) > 0:
            X = X.drop(columns=datetime_cols)
        
        # Update feature count after removing datetime columns
        n_samples, n_features = X.shape
        
        info = DatasetInfo(
            problem_type=problem_type,
            classification_type=clf_type,
            n_samples=n_samples,
            n_features=n_features,
            missing_percentage=missing_pct,
            is_time_series=is_ts
        )
        
        # 3. Simple Preprocessing
        
        # Encode Target if classification and not numeric
        if problem_type == "classification":
            if not pd.api.types.is_numeric_dtype(y) or clf_type is not None: 
                # Force encoding for consistency even if numeric class labels look weird
                self.label_encoder = LabelEncoder()
                y = pd.Series(self.label_encoder.fit_transform(y.astype(str)), index=y.index, name=self.target_col)
        
        # Handle Missing Values (Simple Imputation)
        # Separate numeric and categorical features
        num_cols = X.select_dtypes(include=[np.number]).columns
        # Exclude datetime columns from imputation to avoid SimpleImputer errors
        cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
        
        if len(num_cols) > 0:
            self.num_imputer = SimpleImputer(strategy='median')
            X[num_cols] = self.num_imputer.fit_transform(X[num_cols])
            
        if len(cat_cols) > 0:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols] = self.cat_imputer.fit_transform(X[cat_cols])
        
        # Encode categorical features to numeric values
        # This is essential for sklearn models which require numeric input
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.feature_encoders[col] = le
            
        return X, y, info
