from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .forms import DatasetUploadForm
import os
import sys
import traceback
import pandas as pd

# Add parent directory to path to import ML modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.data_processor import DataProcessor
    from src.meta_features import MetaFeatureExtractor
    from src.meta_learner import MetaLearner, MetaDatasetGenerator
    from src.model_trainer import ModelTrainer
    from src.explainer import ModelExplainer, SystemExplainer
    from src.report_generator import ReportGenerator
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")


def home(request):
    """
    Landing page view.
    """
    return render(request, 'home.html')


def upload_dataset(request):
    """
    Dataset upload view.
    Shows file upload form.
    """
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            try:
                # Get the uploaded file
                uploaded_file = request.FILES['dataset_file']
                
                # Save file to media directory
                fs = FileSystemStorage()
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = fs.path(filename)
                
                # Store file path in session
                request.session['uploaded_file_path'] = file_path
                request.session['uploaded_file_name'] = uploaded_file.name
                
                # Redirect to preview page
                return redirect('preview_dataset')
                
            except Exception as e:
                error_msg = f"Error uploading file: {str(e)}"
                messages.error(request, error_msg)
                form = DatasetUploadForm()
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = DatasetUploadForm()
    
    return render(request, 'upload.html', {'form': form})


def preview_dataset(request):
    """
    Preview dataset and select target column.
    """
    file_path = request.session.get('uploaded_file_path')
    file_name = request.session.get('uploaded_file_name', 'dataset.csv')
    
    if not file_path or not os.path.exists(file_path):
        messages.error(request, 'No dataset found. Please upload a file first.')
        return redirect('upload')
    
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Get dataset info
        n_rows, n_cols = df.shape
        columns = df.columns.tolist()
        
        # Get first 10 rows as HTML table
        preview_html = df.head(10).to_html(classes='table table-striped', index=False)
        
        # Get column info (name, type, non-null count)
        column_info = []
        for col in df.columns:
            column_info.append({
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null': df[col].notna().sum(),
                'null': df[col].isna().sum()
            })
        
        context = {
            'file_name': file_name,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'columns': columns,
            'column_info': column_info,
            'preview_html': preview_html
        }
        
        return render(request, 'preview.html', context)
        
    except Exception as e:
        messages.error(request, f'Error reading dataset: {str(e)}')
        return redirect('upload')


def analyze_dataset(request):
    """
    Analyze dataset with selected target column.
    Runs the complete ML pipeline.
    """
    if request.method != 'POST':
        return redirect('preview_dataset')
    
    file_path = request.session.get('uploaded_file_path')
    file_name = request.session.get('uploaded_file_name', 'dataset.csv')
    target_column = request.POST.get('target_column')
    
    if not file_path or not os.path.exists(file_path):
        messages.error(request, 'No dataset found. Please upload a file first.')
        return redirect('upload')
    
    if not target_column:
        messages.error(request, 'Please select a target column.')
        return redirect('preview_dataset')
    
    try:
        # ===== ML PIPELINE INTEGRATION =====
        
        # Step 1: Process Dataset
        processor = DataProcessor(target_col=target_column)
        X, y, dataset_info = processor.process_and_analyze(file_path)
        
        # Step 2: Extract Meta-Features
        extractor = MetaFeatureExtractor(problem_type=dataset_info.problem_type)
        meta_vector, meta_dict = extractor.compute_all(X, y)
        
        # Step 3 & 4: Select Model using Meta-Learner
        meta_learner_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'meta_learner.pkl')
        
        if os.path.exists(meta_learner_path):
            meta_learner = MetaLearner.load(meta_learner_path)
        else:
            # Train a simple meta-learner on-the-fly
            generator = MetaDatasetGenerator(random_state=42)
            X_meta, y_meta = generator.generate_meta_dataset(n_samples=200)
            meta_learner = MetaLearner(random_state=42)
            meta_learner.train(X_meta, y_meta, cv=3)
            # Save it
            os.makedirs(os.path.dirname(meta_learner_path), exist_ok=True)
            meta_learner.save(meta_learner_path)
        
        selected_model = meta_learner.predict(meta_vector)
        confidence = meta_learner.predict_proba(meta_vector)
        
        # Step 5: Train and Evaluate Selected Model
        trainer = ModelTrainer(selected_model, dataset_info.problem_type, random_state=42)
        result = trainer.train_and_evaluate(X, y, test_size=0.2, tune_hyperparameters=False)
        
        # Generate System Explanation
        meta_importance = meta_learner.get_feature_importance()
        sys_explanation = SystemExplainer.explain_selection(selected_model, meta_dict, meta_importance)
        
        # Model Explainability (optional - can be slow)
        try:
            explainer = ModelExplainer(result.model, X, dataset_info.problem_type)
            shap_results = explainer.explain_with_shap(max_samples=50)
        except Exception as e:
            shap_results = {'error': str(e), 'feature_importance': {}}
        
        # Prepare context for results page
        context = {
            'uploaded_file': file_name,
            'target_column': target_column,
            'dataset_info': dataset_info,
            'meta_features': meta_dict,
            'selected_model': selected_model,
            'confidence': dict(list(confidence.items())[:3]),  # Top 3
            'metrics': result.metrics,
            'best_params': result.best_params,
            'training_time': result.training_time,
            'sys_explanation': sys_explanation,
            'shap_results': shap_results,
        }
        
        # Clear session
        request.session.pop('uploaded_file_path', None)
        request.session.pop('uploaded_file_name', None)
        
        messages.success(request, 'Analysis completed successfully!')
        return render(request, 'results.html', context)
        
    except Exception as e:
        error_msg = f"Error processing dataset: {str(e)}"
        print(f"Full error: {traceback.format_exc()}")
        messages.error(request, error_msg)
        return redirect('preview_dataset')

