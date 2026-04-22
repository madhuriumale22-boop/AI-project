import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "phishing_dataset.csv"
MODEL_PATH = "model.pkl"
ENCODER_PATH = "label_encoder.pkl"

def generate_synthetic_data(num_samples=3000):
    """
    Generates a synthetic dataset for demonstration purposes if a real dataset isn't found.
    Features mimic extract_features:
    url_length, dot_count, dash_count, slash_count, digit_count, subdomain_count,
    contains_at, contains_https, contains_ip, contains_suspicious, uses_short
    Target: 0=Safe, 1=Suspicious, 2=Phishing
    """
    np.random.seed(42)
    print("Generating synthetic dataset...")
    
    data = []
    for _ in range(num_samples):
        # Determine class first to generate appropriate features
        # 0: Safe (40%), 1: Suspicious (30%), 2: Phishing (30%)
        label = np.random.choice([0, 1, 2], p=[0.4, 0.3, 0.3])
        
        if label == 0: # Safe
            url_length = int(np.random.normal(30, 10))
            dot_count = np.random.poisson(2)
            dash_count = np.random.poisson(0.5)
            slash_count = np.random.poisson(3)
            digit_count = np.random.poisson(1)
            subdomain_count = np.random.choice([0, 1], p=[0.8, 0.2])
            contains_at = 0
            contains_https = np.random.choice([0, 1], p=[0.1, 0.9])
            contains_ip = 0
            contains_suspicious = np.random.choice([0, 1], p=[0.95, 0.05])
            uses_short = 0
        elif label == 1: # Suspicious
            url_length = int(np.random.normal(60, 20))
            dot_count = np.random.poisson(3)
            dash_count = np.random.poisson(2)
            slash_count = np.random.poisson(5)
            digit_count = np.random.poisson(5)
            subdomain_count = np.random.choice([1, 2], p=[0.6, 0.4])
            contains_at = np.random.choice([0, 1], p=[0.9, 0.1])
            contains_https = np.random.choice([0, 1], p=[0.5, 0.5])
            contains_ip = np.random.choice([0, 1], p=[0.9, 0.1])
            contains_suspicious = np.random.choice([0, 1], p=[0.6, 0.4])
            uses_short = np.random.choice([0, 1], p=[0.7, 0.3])
        else: # Phishing
            url_length = int(np.random.normal(80, 25))
            dot_count = np.random.poisson(4)
            dash_count = np.random.poisson(4)
            slash_count = np.random.poisson(6)
            digit_count = np.random.poisson(15)
            subdomain_count = np.random.choice([2, 3, 4], p=[0.5, 0.3, 0.2])
            contains_at = np.random.choice([0, 1], p=[0.7, 0.3])
            contains_https = np.random.choice([0, 1], p=[0.8, 0.2]) # Many phishing sites use http
            contains_ip = np.random.choice([0, 1], p=[0.7, 0.3])
            contains_suspicious = np.random.choice([0, 1], p=[0.2, 0.8])
            uses_short = np.random.choice([0, 1], p=[0.6, 0.4])
            
        # Ensure non-negative length
        url_length = max(10, url_length)
        
        data.append([
            url_length, dot_count, dash_count, slash_count, digit_count,
            subdomain_count, contains_at, contains_https, contains_ip,
            contains_suspicious, uses_short, label
        ])
        
    columns = [
        'url_length', 'dot_count', 'dash_count', 'slash_count', 'digit_count',
        'subdomain_count', 'contains_at', 'contains_https', 'contains_ip',
        'contains_suspicious', 'uses_short', 'label'
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Synthetic dataset saved to {DATASET_PATH}")
    return df

def main():
    # 1. Load or Generate Dataset
    if not os.path.exists(DATASET_PATH):
        df = generate_synthetic_data()
    else:
        print(f"Loading dataset from {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH)
        
    # 2. Clean missing values
    if df.isnull().values.any():
        print("Cleaning missing values...")
        df = df.dropna()
        
    # Assume 'label' is the target column. Let's make sure it exists.
    if 'label' not in df.columns:
        # Fallback if a custom CSV was used with different naming
        target_col = df.columns[-1]
        print(f"Column 'label' not found. Using '{target_col}' as target.")
        df.rename(columns={target_col: 'label'}, inplace=True)
        
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Optional: If labels are strings (e.g., 'Safe', 'Phishing'), encode them
    if y.dtype == 'object' or y.dtype.name == 'category':
        print("Encoding labels...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, ENCODER_PATH)
        print(f"Label encoder saved to {ENCODER_PATH}")
    else:
        # If already numeric, we can still save an identity-like or mapping encoder if we want, 
        # but for simplicity we will just rely on the mapping in utils.py.
        # We will save a dummy object just to have the file if needed.
        joblib.dump(None, ENCODER_PATH)
        
    # 3. Split into train/test
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Train RandomForestClassifier
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate and Print metrics
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {acc * 100:.2f}%\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save model.pkl
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    print("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()
