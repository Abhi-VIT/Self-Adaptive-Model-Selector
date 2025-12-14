# User Guide

This guide provides step-by-step instructions on how to use the **Self-Adaptive Model Selector** web application to analyze your datasets.

## Table of Contents
1.  [Getting Started](#getting-started)
2.  [Step 1: Uploading a Dataset](#step-1-uploading-a-dataset)
3.  [Step 2: Preview & Configuration](#step-2-preview--configuration)
4.  [Step 3: Understanding Results](#step-3-understanding-results)
    *   [Model Selection](#model-selection)
    *   [System Explanation](#system-explanation)
    *   [Model Performance](#model-performance)
    *   [Feature Importance (SHAP)](#feature-importance-shap)

---

## Getting Started

Ensure the web server is running:

```bash
# From the project root
cd webapp
python manage.py runserver
```

Open your web browser and navigate to: `http://127.0.0.1:8000/`

You will see the home page:
> "Welcome to Self-Adaptive Model Selector. Intelligent, Explained, Automated AI."

Click the **"Start Analysis"** button to begin.

---

## Step 1: Uploading a Dataset

1.  On the **Upload Dataset** page, you will see a file upload area.
2.  Click **"Choose File"** (or Browse) to select your dataset.
    *   **Accepted Format**: CSV (`.csv`) files.
    *   **Requirements**: The file should contain headers (column names) in the first row.
3.  Click the blue **"Upload & Continue"** button.

*Note: The application currently stores the uploaded file temporarily for the duration of your session.*

---

## Step 2: Preview & Configuration

After uploading, you will be redirected to the **Dataset Preview** page.

1.  **Data Preview**: A table showing the first 10 rows of your dataset allows you to verify the data was read correctly.
2.  **Column Information**: A summary table lists all columns, their data types, and the count of non-null values.
3.  **Select Target Column**:
    *   Locate the "Target Column" dropdown menu at the top or bottom of the page.
    *   Select the column you want the model to predict (e.g., `price`, `species`, `churn`).
4.  Click **"Run Analysis"** to start the ML pipeline.

*Note: Depending on the dataset size, the analysis might take anywhere from a few seconds to a minute.*

---

## Step 3: Understanding Results

The **Analysis Results** page presents a comprehensive report.

### Model Selection
The system displays the **Selected Model** (e.g., "Random Forest Classifier") that it determined is best for your data. It also shows a "Confidence" score for the top 3 candidate models, indicating how sure the system is about its choice.

### System Explanation
This section answers: *"Why was this model chosen?"*
It lists the key reasons based on your dataset's meta-features.
*   *Example: "High non-linearity detected (p-value < 0.05), suggesting tree-based models or neural networks."*
*   *Example: "Dataset has missing values; Random Forest handles them natively."*

### Model Performance
The system splits your data (80% train, 20% test) and trains the selected model.
*   **Classification Metrics**: Accuracy, Precision, Recall, F1-Score.
*   **Regression Metrics**: MSE (Mean Squared Error), R2 Score, MAE.
*   **Best Parameters**: If hyperparameter tuning was run, the optimal settings found are listed here.

### Feature Importance (SHAP)
This interactive visualization helps you understand *how* the model makes predictions.
*   **Bar Chart**: Shows which features had the biggest impact on the predictions globally.
*   The higher the bar, the more important the feature.

---

## Troubleshooting

-   **"Error reading dataset"**: Ensure your CSV file is properly formatted and comma-separated.
-   **"No target column selected"**: You must choose a target variable in Step 2.
-   **Analysis is slow**: Very large datasets (e.g., >100MB) can take time. Support for large-scale data is planned for future versions.
