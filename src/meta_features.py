import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings

@dataclass
class MetaFeatures:
    """Container for computed meta-features."""
    dimensionality_ratio: float = np.nan
    avg_vif: float = np.nan
    breusch_pagan_pvalue: float = np.nan
    box_tidwell_pvalue: float = np.nan
    mean_skewness: float = np.nan
    mean_kurtosis: float = np.nan
    outlier_percentage: float = np.nan
    class_imbalance_ratio: float = np.nan
    
    def to_dict(self) -> Dict[str, float]:
        """Return as labeled dictionary."""
        return asdict(self)
    
    def to_vector(self) -> np.ndarray:
        """Return as numeric vector."""
        return np.array([
            self.dimensionality_ratio,
            self.avg_vif,
            self.breusch_pagan_pvalue,
            self.box_tidwell_pvalue,
            self.mean_skewness,
            self.mean_kurtosis,
            self.outlier_percentage,
            self.class_imbalance_ratio
        ])


class MetaFeatureExtractor:
    """
    Computes statistical and data characteristics as meta-features.
    
    Meta-features include:
    - Dimensionality ratio (features/samples)
    - Average VIF (Variance Inflation Factor)
    - Breusch-Pagan test p-value for heteroscedasticity
    - Box-Tidwell test p-value for non-linearity
    - Mean skewness and kurtosis
    - Outlier percentage (IQR method)
    - Class imbalance ratio (classification only)
    """
    
    def __init__(self, problem_type: str = "regression"):
        """
        Initialize the meta-feature extractor.
        
        Args:
            problem_type: "regression" or "classification"
        """
        self.problem_type = problem_type
        
    def compute_all(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute all applicable meta-features.
        
        Args:
            X: Feature matrix (preprocessed)
            y: Target variable
            
        Returns:
            Tuple of (numeric_vector, labeled_dict)
        """
        meta = MetaFeatures()
        
        # Always compute these
        meta.dimensionality_ratio = self._dimensionality_ratio(X)
        meta.mean_skewness, meta.mean_kurtosis = self._skewness_kurtosis(X)
        meta.outlier_percentage = self._outlier_percentage(X)
        
        # VIF - may fail for singular matrices
        try:
            meta.avg_vif = self._average_vif(X)
        except Exception as e:
            warnings.warn(f"VIF computation failed: {e}")
            meta.avg_vif = np.nan
        
        # Regression-specific tests
        if self.problem_type == "regression":
            try:
                meta.breusch_pagan_pvalue = self._breusch_pagan_test(X, y)
            except Exception as e:
                warnings.warn(f"Breusch-Pagan test failed: {e}")
                meta.breusch_pagan_pvalue = np.nan
                
            try:
                meta.box_tidwell_pvalue = self._box_tidwell_test(X, y)
            except Exception as e:
                warnings.warn(f"Box-Tidwell test failed: {e}")
                meta.box_tidwell_pvalue = np.nan
        
        # Classification-specific
        if self.problem_type == "classification":
            try:
                meta.class_imbalance_ratio = self._class_imbalance_ratio(y)
            except Exception as e:
                warnings.warn(f"Class imbalance computation failed: {e}")
                meta.class_imbalance_ratio = np.nan
        
        return meta.to_vector(), meta.to_dict()
    
    def _dimensionality_ratio(self, X: pd.DataFrame) -> float:
        """Compute ratio of features to samples."""
        n_samples, n_features = X.shape
        if n_samples == 0:
            return np.nan
        return n_features / n_samples
    
    def _average_vif(self, X: pd.DataFrame) -> float:
        """
        Compute average Variance Inflation Factor.
        
        VIF measures multicollinearity by regressing each feature
        against all others.
        """
        # Only use numeric columns
        X_num = X.select_dtypes(include=[np.number])
        
        if X_num.shape[1] < 2:
            return np.nan
        
        # Convert to numpy for faster computation
        X_arr = X_num.values
        n_features = X_arr.shape[1]
        
        vif_values = []
        for i in range(n_features):
            # Skip constant features
            if np.std(X_arr[:, i]) == 0:
                continue
                
            # Get feature i and all others
            y_i = X_arr[:, i]
            X_others = np.delete(X_arr, i, axis=1)
            
            # Check if X_others has variation
            if X_others.shape[1] == 0 or np.all(np.std(X_others, axis=0) == 0):
                continue
            
            try:
                # Compute R² using OLS
                # Add intercept
                X_with_intercept = np.column_stack([np.ones(len(X_others)), X_others])
                
                # Solve least squares
                coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y_i, rcond=None)
                
                # Compute R²
                y_pred = X_with_intercept @ coeffs
                ss_res = np.sum((y_i - y_pred) ** 2)
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                
                if ss_tot == 0:
                    continue
                    
                r_squared = 1 - (ss_res / ss_tot)
                
                # VIF = 1 / (1 - R²)
                if r_squared >= 0.9999:  # Near perfect collinearity
                    vif = 1000  # Cap at large value
                else:
                    vif = 1 / (1 - r_squared)
                    
                vif_values.append(vif)
            except:
                continue
        
        if len(vif_values) == 0:
            return np.nan
        
        return np.mean(vif_values)
    
    def _breusch_pagan_test(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Breusch-Pagan test for heteroscedasticity.
        
        Returns the p-value. Low p-value indicates heteroscedasticity.
        """
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
            
            # Only use numeric features
            X_num = X.select_dtypes(include=[np.number])
            
            if X_num.shape[1] == 0:
                return np.nan
            
            # Add constant
            X_with_const = sm.add_constant(X_num)
            
            # Fit OLS
            model = OLS(y, X_with_const).fit()
            
            # Breusch-Pagan test
            bp_test = het_breuschpagan(model.resid, X_with_const)
            
            # Return p-value (index 1)
            return bp_test[1]
            
        except Exception as e:
            raise e
    
    def _box_tidwell_test(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Box-Tidwell test for non-linearity.
        
        Tests if features need non-linear transformations.
        Returns the minimum p-value across features.
        """
        try:
            import statsmodels.api as sm
            from statsmodels.regression.linear_model import OLS
            
            # Only use numeric features
            X_num = X.select_dtypes(include=[np.number])
            
            if X_num.shape[1] == 0:
                return np.nan
            
            p_values = []
            
            for col in X_num.columns:
                x_col = X_num[col].values
                
                # Skip if constant or has non-positive values (log not applicable)
                if np.std(x_col) == 0 or np.any(x_col <= 0):
                    continue
                
                try:
                    # Box-Tidwell: regress y on x and x*log(x)
                    x_log = x_col * np.log(x_col)
                    
                    # Create design matrix with intercept
                    X_design = sm.add_constant(np.column_stack([x_col, x_log]))
                    
                    # Fit model
                    model = OLS(y, X_design).fit()
                    
                    # P-value for x*log(x) term (last coefficient)
                    p_val = model.pvalues[-1]
                    p_values.append(p_val)
                    
                except:
                    continue
            
            if len(p_values) == 0:
                return np.nan
            
            # Return minimum p-value (most significant non-linearity)
            return np.min(p_values)
            
        except Exception as e:
            raise e
    
    def _skewness_kurtosis(self, X: pd.DataFrame) -> Tuple[float, float]:
        """Compute mean skewness and kurtosis across numeric features."""
        X_num = X.select_dtypes(include=[np.number])
        
        if X_num.shape[1] == 0:
            return np.nan, np.nan
        
        skewness_values = []
        kurtosis_values = []
        
        for col in X_num.columns:
            col_data = X_num[col].dropna()
            
            if len(col_data) < 3:  # Need at least 3 points
                continue
            
            if np.std(col_data) == 0:  # Skip constant features
                continue
            
            skewness_values.append(skew(col_data))
            kurtosis_values.append(kurtosis(col_data))
        
        mean_skew = np.mean(skewness_values) if len(skewness_values) > 0 else np.nan
        mean_kurt = np.mean(kurtosis_values) if len(kurtosis_values) > 0 else np.nan
        
        return mean_skew, mean_kurt
    
    def _outlier_percentage(self, X: pd.DataFrame) -> float:
        """
        Compute percentage of outliers using IQR method.
        
        Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
        """
        X_num = X.select_dtypes(include=[np.number])
        
        if X_num.shape[1] == 0:
            return np.nan
        
        total_values = 0
        outlier_count = 0
        
        for col in X_num.columns:
            col_data = X_num[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
            outlier_count += outliers
            total_values += len(col_data)
        
        if total_values == 0:
            return np.nan
        
        return (outlier_count / total_values) * 100
    
    def _class_imbalance_ratio(self, y: pd.Series) -> float:
        """
        Compute class imbalance ratio.
        
        Returns minority class size / majority class size.
        Value close to 0 indicates high imbalance.
        """
        class_counts = y.value_counts()
        
        if len(class_counts) < 2:
            return np.nan
        
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        
        if max_class_size == 0:
            return np.nan
        
        return min_class_size / max_class_size
