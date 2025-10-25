"""
Model Trainer for AI Disease Prediction System
Trains a Random Forest classifier on the synthetic dataset and evaluates performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class DiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.label_encoder = LabelEncoder()
        self.symptoms = []
        
    def load_data(self, filepath):
        """Load the dataset from CSV file."""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {len(df)} records")
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training."""
        print("Preprocessing data...")
        
        # Separate features and target
        self.symptoms = [col for col in df.columns if col != 'disease']
        X = df[self.symptoms]
        y = df['disease']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features: {len(self.symptoms)} symptoms")
        print(f"Classes: {len(self.label_encoder.classes_)} diseases")
        
        return X, y_encoded, y
    
    def train_model(self, X, y):
        """Train the Random Forest model."""
        print("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def evaluate_model(self, X_test, y_test, y_pred, y_original):
        """Evaluate model performance."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return accuracy, cv_scores
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix."""
        print("Generating confusion matrix...")
        
        # Convert back to disease names
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Disease Prediction')
        plt.xlabel('Predicted Disease')
        plt.ylabel('Actual Disease')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self):
        """Plot feature importance."""
        print("Generating feature importance plot...")
        
        # Get feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'symptom': self.symptoms,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='symptom')
        plt.title('Top 15 Most Important Symptoms for Disease Prediction')
        plt.xlabel('Feature Importance')
        plt.ylabel('Symptoms')
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def save_model(self):
        """Save the trained model and label encoder."""
        print("Saving model and encoders...")
        
        # Create models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, 'disease_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save label encoder
        encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save symptoms list
        symptoms_path = os.path.join(models_dir, 'symptoms.pkl')
        joblib.dump(self.symptoms, symptoms_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Label encoder saved to: {encoder_path}")
        print(f"Symptoms list saved to: {symptoms_path}")
        
        return model_path, encoder_path, symptoms_path
    
    def predict_disease(self, symptoms_input):
        """Predict disease based on symptoms."""
        # Create input vector
        input_vector = np.zeros(len(self.symptoms))
        for i, symptom in enumerate(self.symptoms):
            if symptom in symptoms_input:
                input_vector[i] = 1
        
        # Make prediction
        prediction = self.model.predict([input_vector])[0]
        probability = self.model.predict_proba([input_vector])[0]
        
        # Get disease name
        disease = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probability[prediction]
        
        return disease, confidence

def main():
    """Main function to train and evaluate the model."""
    print("AI Disease Prediction System - Model Trainer")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DiseasePredictor()
    
    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'disease_dataset.csv')
    df = predictor.load_data(data_path)
    
    # Preprocess data
    X, y_encoded, y_original = predictor.preprocess_data(df)
    
    # Train model
    X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y_encoded)
    
    # Evaluate model
    accuracy, cv_scores = predictor.evaluate_model(X_test, y_test, y_pred, y_original)
    
    # Generate plots
    predictor.plot_confusion_matrix(y_test, y_pred)
    feature_importance = predictor.plot_feature_importance()
    
    # Save model
    model_path, encoder_path, symptoms_path = predictor.save_model()
    
    # Test prediction
    print("\n" + "="*50)
    print("TESTING PREDICTION")
    print("="*50)
    test_symptoms = ['fever', 'cough', 'fatigue', 'body_ache']
    disease, confidence = predictor.predict_disease(test_symptoms)
    print(f"Test symptoms: {test_symptoms}")
    print(f"Predicted disease: {disease}")
    print(f"Confidence: {confidence:.4f}")
    
    print(f"\n✅ Model training completed successfully!")
    print(f"📊 Model accuracy: {accuracy:.4f}")
    print(f"💾 Model saved and ready for use!")

if __name__ == "__main__":
    main()