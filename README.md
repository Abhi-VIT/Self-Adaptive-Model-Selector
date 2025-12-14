# Self-Adaptive Model Selector

A comprehensive, automated Machine Learning pipeline that intelligently selects, trains, and explains the best model for your dataset. This system analyzes your data's characteristics (meta-features) to choose the optimal algorithm, ensuring high performance without the need for manual model selection.

## 🚀 Key Features

*   **Intelligent Model Selection**: Uses a Meta-Learner (trained on dataset characteristics) to predict the best performing model type (e.g., Random Forest, XGBoost, SVM, Linear).
*   **Automatic Statistics extraction**: Automatically detects problem type (Classification/Regression), handles missing values, and extracts statistical meta-features (Skewness, Kurtosis, VIF, etc.).
*   **Explainable AI (XAI)**:
    *   **System Explanation**: Explains *why* a specific model was chosen based on dataset properties.
    *   **Model Explanation**: Uses SHAP (SHapley Additive exPlanations) values to explain individual predictions.
*   **User-Friendly Web Interface**: A clean, modern Django web application to upload datasets, visualize data, and view analysis reports.
*   **Robust ML Pipeline**: Includes data preprocessing, meta-feature extraction, model training, and performance evaluation.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Self-Adaptive-Model-Selector
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### Running the Web Application

The easiest way to use the system is via the web interface.

1.  **Navigate to the webapp directory:**
    ```bash
    cd webapp
    ```

2.  **Apply database migrations:**
    ```bash
    python manage.py migrate
    ```

3.  **Start the development server:**
    ```bash
    python manage.py runserver
    ```

4.  **Access the app:**
    Open your browser and go to `http://127.0.0.1:8000/`.

### Using the ML Pipeline Directly

You can also use the source code directly in your Python scripts:

```python
import pandas as pd
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

# 1. Load and Process Data
processor = DataProcessor(target_col='target')
X, y, info = processor.process_and_analyze('your_dataset.csv')

# 2. Train a specific model (e.g., Random Forest)
trainer = ModelTrainer("Random Forest Classifier", info.problem_type)
result = trainer.train_and_evaluate(X, y)

print(f"Accuracy: {result.metrics['accuracy']}")
```

## 📂 Project Structure

```
Self-Adaptive-Model-Selector/
├── docs/               # Detailed documentation
├── models/             # Saved trained models (e.g., meta_learner.pkl)
├── src/                # Core ML source code
│   ├── data_processor.py   # Data loading & preprocessing
│   ├── meta_features.py    # Statistical feature extraction
│   ├── meta_learner.py     # Model selection logic
│   ├── model_trainer.py    # Training & evaluation
│   ├── explainer.py        # SHAP & System explanations
│   └── ...
├── webapp/             # Django Web Application
│   ├── manage.py       # Django CLI entry point
│   ├── selector/       # Main app ("views", "urls", "templates")
│   └── ...
├── tests/              # Unit and integration tests
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## 📚 Documentation

For more in-depth details, check out the documentation folder:

*   [**User Guide**](docs/USER_GUIDE.md): Step-by-step instructions for using the web app.
*   [**Architecture Overview**](docs/ARCHITECTURE.md): Deep dive into the system design and data flow.
*   [**API Reference**](docs/API_REFERENCE.md): Detailed class and function documentation for developers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT License](LICENSE) (or your preferred license)
