from django import forms
from django.core.exceptions import ValidationError
import os


class DatasetUploadForm(forms.Form):
    """
    Form for uploading CSV dataset files.
    """
    dataset_file = forms.FileField(
        label='Select Dataset',
        help_text='Upload a CSV file (max 10MB)',
        widget=forms.FileInput(attrs={
            'class': 'file-input',
            'accept': '.csv'
        })
    )
    
    def clean_dataset_file(self):
        """
        Validate uploaded file.
        """
        file = self.cleaned_data.get('dataset_file')
        
        if not file:
            raise ValidationError("No file was uploaded.")
        
        # Check file extension
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext != '.csv':
            raise ValidationError("Only CSV files are allowed.")
        
        # Check file size (10MB = 10485760 bytes)
        if file.size > 10485760:
            raise ValidationError("File size must be less than 10MB.")
        
        return file
