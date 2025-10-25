"""
Data Generator for AI Disease Prediction System
Creates a synthetic dataset with symptoms and diseases for training the ML model.
"""

import pandas as pd
import numpy as np
import random
import os

# Define symptoms and diseases
SYMPTOMS = [
    'fever', 'cough', 'headache', 'fatigue', 'sore_throat', 'runny_nose',
    'body_ache', 'nausea', 'vomiting', 'diarrhea', 'chest_pain', 'shortness_of_breath',
    'dizziness', 'skin_rash', 'joint_pain', 'loss_of_appetite', 'weight_loss',
    'abdominal_pain', 'back_pain', 'difficulty_swallowing'
]

DISEASES = [
    'Common Cold', 'Flu', 'Migraine', 'Food Poisoning', 'Gastroenteritis',
    'Pneumonia', 'Bronchitis', 'Allergic Reaction', 'Muscle Strain', 'Hypertension'
]

# Define disease-symptom relationships (probability of symptom occurring with disease)
DISEASE_SYMPTOM_MAPPING = {
    'Common Cold': {
        'runny_nose': 0.9, 'sore_throat': 0.8, 'cough': 0.7, 'fatigue': 0.6,
        'headache': 0.4, 'fever': 0.3, 'body_ache': 0.3
    },
    'Flu': {
        'fever': 0.9, 'body_ache': 0.8, 'fatigue': 0.9, 'headache': 0.7,
        'cough': 0.6, 'sore_throat': 0.5, 'nausea': 0.4, 'loss_of_appetite': 0.6
    },
    'Migraine': {
        'headache': 1.0, 'nausea': 0.7, 'vomiting': 0.4, 'dizziness': 0.6,
        'fatigue': 0.5, 'difficulty_swallowing': 0.2
    },
    'Food Poisoning': {
        'nausea': 0.9, 'vomiting': 0.8, 'diarrhea': 0.9, 'abdominal_pain': 0.8,
        'fever': 0.5, 'fatigue': 0.7, 'loss_of_appetite': 0.8
    },
    'Gastroenteritis': {
        'diarrhea': 0.9, 'nausea': 0.8, 'vomiting': 0.7, 'abdominal_pain': 0.8,
        'fever': 0.6, 'fatigue': 0.7, 'loss_of_appetite': 0.6
    },
    'Pneumonia': {
        'cough': 0.9, 'fever': 0.8, 'shortness_of_breath': 0.8, 'chest_pain': 0.7,
        'fatigue': 0.8, 'body_ache': 0.6, 'headache': 0.5
    },
    'Bronchitis': {
        'cough': 0.9, 'chest_pain': 0.6, 'shortness_of_breath': 0.5, 'fatigue': 0.7,
        'fever': 0.4, 'body_ache': 0.4
    },
    'Allergic Reaction': {
        'skin_rash': 0.8, 'runny_nose': 0.6, 'cough': 0.5, 'shortness_of_breath': 0.4,
        'dizziness': 0.3, 'nausea': 0.3
    },
    'Muscle Strain': {
        'body_ache': 0.9, 'joint_pain': 0.7, 'back_pain': 0.8, 'fatigue': 0.5,
        'headache': 0.3
    },
    'Hypertension': {
        'headache': 0.6, 'dizziness': 0.7, 'chest_pain': 0.4, 'shortness_of_breath': 0.3,
        'fatigue': 0.5, 'nausea': 0.3
    }
}

def generate_patient_record(disease):
    """Generate a single patient record with symptoms based on disease."""
    record = {'disease': disease}
    
    # Initialize all symptoms as 0
    for symptom in SYMPTOMS:
        record[symptom] = 0
    
    # Set symptoms based on disease mapping
    disease_symptoms = DISEASE_SYMPTOM_MAPPING.get(disease, {})
    
    for symptom, probability in disease_symptoms.items():
        if random.random() < probability:
            record[symptom] = 1
    
    # Add some random noise (false positives/negatives)
    for symptom in SYMPTOMS:
        if symptom not in disease_symptoms:
            if random.random() < 0.05:  # 5% chance of random symptom
                record[symptom] = 1
        else:
            if record[symptom] == 1 and random.random() < 0.1:  # 10% chance of missing expected symptom
                record[symptom] = 0
    
    return record

def generate_dataset(num_records=1200):
    """Generate the complete dataset."""
    print("Generating synthetic disease dataset...")
    
    records = []
    
    # Generate records for each disease
    records_per_disease = num_records // len(DISEASES)
    
    for disease in DISEASES:
        for _ in range(records_per_disease):
            record = generate_patient_record(disease)
            records.append(record)
    
    # Generate additional random records to reach target number
    remaining_records = num_records - len(records)
    for _ in range(remaining_records):
        disease = random.choice(DISEASES)
        record = generate_patient_record(disease)
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def save_dataset(df, filename='disease_dataset.csv'):
    """Save the dataset to CSV file."""
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")
    
    return filepath

def display_dataset_info(df):
    """Display information about the generated dataset."""
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Total symptoms: {len(SYMPTOMS)}")
    print(f"Total diseases: {len(DISEASES)}")
    
    print("\nDisease distribution:")
    disease_counts = df['disease'].value_counts()
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count} records")
    
    print("\nSymptom frequency:")
    symptom_freq = df[SYMPTOMS].sum().sort_values(ascending=False)
    for symptom, freq in symptom_freq.head(10).items():
        print(f"  {symptom}: {freq} occurrences")
    
    print("\nFirst 5 records:")
    print(df.head())

def main():
    """Main function to generate and save the dataset."""
    print("AI Disease Prediction System - Data Generator")
    print("=" * 50)
    
    # Generate dataset
    df = generate_dataset(num_records=1200)
    
    # Display information
    display_dataset_info(df)
    
    # Save dataset
    filepath = save_dataset(df)
    
    print(f"\n✅ Dataset generation completed successfully!")
    print(f"📁 Dataset saved at: {filepath}")
    print(f"📊 Ready for model training!")

if __name__ == "__main__":
    main()