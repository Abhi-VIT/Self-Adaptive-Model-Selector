# API Reference

This reference documents the core Python modules extracted from `src/`.

## `src.data_processor`

### `class DataProcessor`
Handles data loading and preprocessing.

#### `__init__(self, target_col: str)`
*   `target_col`: Name of the target variable column.

#### `process_and_analyze(self, source: Union[str, pd.DataFrame])`
Runs the full processing pipeline.
*   **Args**:
    *   `source`: File path or DataFrame.
*   **Returns**: `(X, y, info)`
    *   `X`: Processed feature DataFrame.
    *   `y`: Target Series.
    *   `info`: `DatasetInfo` object.

---

## `src.meta_features`

### `class MetaFeatureExtractor`
Computes statistical meta-features.

#### `compute_all(self, X: pd.DataFrame, y: pd.Series) -> Tuple`
Computes all available meta-features.
*   **Returns**: `(vector, dictionary)`
    *   `vector`: Numpy array of values (for model input).
    *   `dictionary`: Key-value pairs of feature names and values.

---

## `src.meta_learner`

### `class MetaLearner`
Predicts the best model algorithm.

#### `predict(self, meta_features: np.ndarray) -> str`
*   **Args**:
    *   `meta_features`: Array of computed meta-features.
*   **Returns**: Name of the predicted model (e.g., "Random Forest Classifier").

---

## `src.model_trainer`

### `class ModelTrainer`
Trains and evaluates models.

#### `train_and_evaluate(self, X, y, test_size=0.2, tune_hyperparameters=True)`
*   **Args**:
    *   `X`: Features.
    *   `y`: Target.
    *   `test_size`: Split ratio (default 0.2).
    *   `tune_hyperparameters`: Boolean to enable/disable grid search.
*   **Returns**: `TrainingResult` object containing:
    *   `model`: The trained model object.
    *   `metrics`: Dictionary of performance scores (Accuracy, MSE, etc.).
    *   `best_params`: Dictionary of best hyperparameters found.

---

## `src.explainer`

### `class SystemExplainer`

#### `explain_selection(selected_model, meta_features, meta_importance) -> List[str]`
Generates natural language explanations for *why* a model was selected.
*   **Returns**: List of explanation strings.

### `class ModelExplainer`

#### `explain_with_shap(max_samples=100)`
Generates SHAP values for local/global explanation.
*   **Returns**: Dictionary containing SHAP values and feature names.
