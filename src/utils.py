"""
Utility functions for the AI Disease Prediction System.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def check_model_files():
    """Check if all required model files exist."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    required_files = ['disease_model.pkl', 'label_encoder.pkl', 'symptoms.pkl']
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def get_project_info():
    """Get project information and statistics."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    info = {
        'project_name': 'AI Disease Prediction System',
        'version': '1.0.0',
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'project_path': project_root
    }
    
    # Check dataset
    data_path = os.path.join(project_root, 'data', 'disease_dataset.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        info['dataset_records'] = len(df)
        info['dataset_diseases'] = df['disease'].nunique()
        info['dataset_symptoms'] = len([col for col in df.columns if col != 'disease'])
    else:
        info['dataset_records'] = 0
        info['dataset_diseases'] = 0
        info['dataset_symptoms'] = 0
    
    # Check model files
    model_exists, missing_files = check_model_files()
    info['model_trained'] = model_exists
    info['missing_files'] = missing_files
    
    return info

def format_symptom_display(symptom_list):
    """Format symptom list for display."""
    return [symptom.replace('_', ' ').title() for symptom in symptom_list]

def validate_symptoms(symptoms, valid_symptoms):
    """Validate that provided symptoms are in the valid list."""
    invalid_symptoms = [s for s in symptoms if s not in valid_symptoms]
    return len(invalid_symptoms) == 0, invalid_symptoms

def calculate_symptom_similarity(symptoms1, symptoms2):
    """Calculate similarity between two symptom sets."""
    set1 = set(symptoms1)
    set2 = set(symptoms2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def get_system_status():
    """Get overall system status."""
    status = {
        'dataset_ready': False,
        'model_ready': False,
        'system_ready': False,
        'messages': []
    }
    
    # Check dataset
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, 'data', 'disease_dataset.csv')
    
    if os.path.exists(data_path):
        status['dataset_ready'] = True
        status['messages'].append("✅ Dataset is available")
    else:
        status['messages'].append("❌ Dataset not found. Run data_generator.py")
    
    # Check model
    model_exists, missing_files = check_model_files()
    if model_exists:
        status['model_ready'] = True
        status['messages'].append("✅ Model is trained and ready")
    else:
        status['messages'].append(f"❌ Model files missing: {', '.join(missing_files)}")
        status['messages'].append("Run model_trainer.py to train the model")
    
    # Overall status
    status['system_ready'] = status['dataset_ready'] and status['model_ready']
    
    if status['system_ready']:
        status['messages'].append("🎉 System is ready for predictions!")
    
    return status

def log_prediction(symptoms, prediction, confidence, timestamp=None):
    """Log prediction for analysis (optional feature)."""
    if timestamp is None:
        timestamp = datetime.now()
    
    log_entry = {
        'timestamp': timestamp.isoformat(),
        'symptoms': symptoms,
        'prediction': prediction,
        'confidence': confidence
    }
    
    return log_entry

def get_disease_statistics(dataset_path=None):
    """Get statistics about diseases in the dataset."""
    if dataset_path is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_path = os.path.join(project_root, 'data', 'disease_dataset.csv')
    
    if not os.path.exists(dataset_path):
        return None
    
    df = pd.read_csv(dataset_path)
    
    stats = {
        'total_records': len(df),
        'disease_distribution': df['disease'].value_counts().to_dict(),
        'symptom_frequency': {},
        'most_common_disease': df['disease'].mode().iloc[0],
        'least_common_disease': df['disease'].value_counts().idxmin()
    }
    
    # Calculate symptom frequency
    symptom_cols = [col for col in df.columns if col != 'disease']
    for symptom in symptom_cols:
        stats['symptom_frequency'][symptom] = df[symptom].sum()
    
    return stats