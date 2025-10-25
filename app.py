"""
AI Disease Prediction System - Streamlit Web Application
Main web interface for the disease prediction system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.predictor import DiseasePredictor
import os

# Page configuration
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = DiseasePredictor()
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'selected_symptoms' not in st.session_state:
        st.session_state.selected_symptoms = []

def format_symptom_name(symptom):
    """Format symptom name for display."""
    return symptom.replace('_', ' ').title()

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox("Choose a page:", ["Disease Prediction", "About System", "Model Information"])
        
        st.markdown("---")
        st.markdown("### ⚠️ Important Disclaimer")
        st.warning("This system is for educational purposes only. Always consult healthcare professionals for medical advice.")
    
    if page == "Disease Prediction":
        prediction_page()
    elif page == "About System":
        about_page()
    elif page == "Model Information":
        model_info_page()

def prediction_page():
    """Disease prediction page."""
    st.header("🔍 Disease Prediction")
    
    # Check if model is loaded
    if not st.session_state.predictor.model_loaded:
        with st.spinner("Loading AI model..."):
            if not st.session_state.predictor.load_model():
                st.error("❌ Failed to load the AI model. Please ensure the model is trained first.")
                st.info("Run `python src/model_trainer.py` to train the model.")
                return
    
    # Get symptoms list
    symptoms = st.session_state.predictor.get_symptoms_list()
    
    if not symptoms:
        st.error("❌ Could not load symptoms list.")
        return
    
    # Symptoms selection
    st.subheader("Select Your Symptoms")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Common Symptoms:**")
        common_symptoms = ['fever', 'cough', 'headache', 'fatigue', 'sore_throat', 'runny_nose', 'body_ache', 'nausea', 'vomiting', 'diarrhea']
        selected_common = []
        for symptom in common_symptoms:
            if symptom in symptoms:
                if st.checkbox(format_symptom_name(symptom), key=f"common_{symptom}"):
                    selected_common.append(symptom)
    
    with col2:
        st.markdown("**Other Symptoms:**")
        other_symptoms = [s for s in symptoms if s not in common_symptoms]
        selected_other = []
        for symptom in other_symptoms:
            if st.checkbox(format_symptom_name(symptom), key=f"other_{symptom}"):
                selected_other.append(symptom)
    
    # Combine selected symptoms
    selected_symptoms = selected_common + selected_other
    
    # Display selected symptoms
    if selected_symptoms:
        st.markdown("### Selected Symptoms:")
        symptom_tags = " | ".join([format_symptom_name(s) for s in selected_symptoms])
        st.info(f"🏷️ {symptom_tags}")
    
    # Prediction button
    if st.button("🔮 Predict Disease", type="primary", disabled=len(selected_symptoms) == 0):
        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing symptoms..."):
                disease, confidence, top_predictions = st.session_state.predictor.predict_disease(selected_symptoms)
                
                if disease:
                    st.session_state.prediction_made = True
                    st.session_state.selected_symptoms = selected_symptoms
                    
                    # Display main prediction
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### 🎯 Primary Prediction")
                    st.markdown(f"**Disease:** {disease}")
                    st.markdown(f"**Confidence:** {confidence:.2%}")
                    
                    # Confidence meter
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Level (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.markdown("### 📊 Top 3 Possible Conditions")
                    
                    for i, (pred_disease, pred_conf) in enumerate(top_predictions):
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col1:
                            st.markdown(f"**#{i+1}**")
                        with col2:
                            st.markdown(f"**{pred_disease}**")
                        with col3:
                            st.markdown(f"{pred_conf:.2%}")
                    
                    # Disease information
                    disease_info = st.session_state.predictor.get_disease_info(disease)
                    
                    st.markdown("### 📖 Disease Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Description:**")
                        st.write(disease_info.get('description', 'No description available.'))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Common Symptoms:**")
                        common_symp = disease_info.get('common_symptoms', [])
                        if common_symp:
                            for symptom in common_symp:
                                st.write(f"• {format_symptom_name(symptom)}")
                        else:
                            st.write("No specific symptoms listed.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("**Recommendations:**")
                        recommendations = disease_info.get('recommendations', [])
                        if recommendations:
                            for rec in recommendations:
                                st.write(f"• {rec}")
                        else:
                            st.write("No specific recommendations available.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Warning
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("⚠️ **Important:** This prediction is based on AI analysis and should not replace professional medical diagnosis. Please consult a healthcare provider for proper medical advice.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.error("❌ Failed to make prediction. Please try again.")

def about_page():
    """About system page."""
    st.header("📚 About the AI Disease Prediction System")
    
    st.markdown("""
    ## 🎯 Overview
    
    The AI Disease Prediction System is an educational tool that demonstrates how machine learning 
    can be applied to healthcare for symptom-based disease prediction. This system uses a 
    Random Forest classifier trained on synthetic medical data to predict possible diseases 
    based on user-reported symptoms.
    
    ## 🔬 How It Works
    
    1. **Data Collection**: The system uses a synthetic dataset containing symptom-disease relationships
    2. **Machine Learning**: A Random Forest classifier is trained to recognize patterns between symptoms and diseases
    3. **Prediction**: When you input symptoms, the AI analyzes the pattern and predicts the most likely diseases
    4. **Confidence Scoring**: Each prediction comes with a confidence score indicating the model's certainty
    
    ## 🎯 Features
    
    - **20 Different Symptoms**: Covers common symptoms like fever, cough, headache, etc.
    - **10 Disease Categories**: Includes common conditions like flu, cold, migraine, etc.
    - **Real-time Predictions**: Instant analysis of symptom combinations
    - **Confidence Scoring**: Transparency in prediction reliability
    - **Disease Information**: Detailed information about predicted conditions
    - **User-friendly Interface**: Easy-to-use web interface built with Streamlit
    
    ## 🛠️ Technology Stack
    
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning library for model training
    - **Streamlit**: Web application framework
    - **Pandas**: Data manipulation and analysis
    - **Plotly**: Interactive visualizations
    - **Random Forest**: Machine learning algorithm for classification
    
    ## ⚠️ Important Limitations
    
    - **Educational Purpose Only**: This system is designed for learning and demonstration
    - **Synthetic Data**: Uses artificially generated data, not real medical records
    - **Limited Scope**: Covers only 10 common conditions and 20 symptoms
    - **Not a Medical Device**: Should never be used for actual medical diagnosis
    - **Requires Professional Consultation**: Always consult healthcare providers for medical concerns
    
    ## 🔮 Future Enhancements
    
    - Integration with real medical datasets (with proper permissions)
    - Support for more symptoms and diseases
    - Advanced ML models (Deep Learning, Ensemble methods)
    - Mobile application development
    - Integration with wearable devices
    - Multi-language support
    """)

def model_info_page():
    """Model information page."""
    st.header("🤖 Model Information")
    
    # Check if model files exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_exists = os.path.exists(os.path.join(models_dir, 'disease_model.pkl'))
    
    if not model_exists:
        st.warning("⚠️ Model not found. Please train the model first by running `python src/model_trainer.py`")
        return
    
    st.markdown("""
    ## 📊 Model Architecture
    
    **Algorithm**: Random Forest Classifier
    
    **Key Parameters**:
    - Number of Estimators: 100
    - Max Depth: 10
    - Min Samples Split: 5
    - Min Samples Leaf: 2
    - Random State: 42 (for reproducibility)
    
    ## 📈 Performance Metrics
    
    The model has been trained and evaluated on a synthetic dataset with the following characteristics:
    
    - **Dataset Size**: 1,200 records
    - **Features**: 20 symptoms (binary encoded)
    - **Classes**: 10 diseases
    - **Train/Test Split**: 80/20
    - **Cross-validation**: 5-fold
    
    ## 🎯 Model Accuracy
    
    Based on the training results:
    - **Test Accuracy**: ~70%
    - **Cross-validation Score**: ~66%
    
    ## 📋 Supported Diseases
    
    The model can predict the following conditions:
    
    1. **Common Cold** - Viral upper respiratory infection
    2. **Flu** - Influenza virus infection
    3. **Migraine** - Severe headache disorder
    4. **Food Poisoning** - Foodborne illness
    5. **Gastroenteritis** - Stomach and intestine inflammation
    6. **Pneumonia** - Lung infection
    7. **Bronchitis** - Bronchial tube inflammation
    8. **Allergic Reaction** - Immune system response
    9. **Muscle Strain** - Muscle or tendon injury
    10. **Hypertension** - High blood pressure
    
    ## 🔍 Feature Importance
    
    The most important symptoms for disease prediction (based on model analysis):
    
    1. Disease-specific symptoms (e.g., runny nose for cold)
    2. Fever (common across many conditions)
    3. Fatigue (general indicator of illness)
    4. Cough (respiratory symptom)
    5. Headache (neurological symptom)
    
    ## ⚙️ Model Training Process
    
    1. **Data Generation**: Synthetic dataset created with realistic symptom-disease relationships
    2. **Preprocessing**: Binary encoding of symptoms, label encoding of diseases
    3. **Training**: Random Forest classifier trained with cross-validation
    4. **Evaluation**: Performance assessed using multiple metrics
    5. **Serialization**: Model saved using joblib for deployment
    
    ## 🔄 Model Updates
    
    To retrain the model with new data:
    ```bash
    python src/model_trainer.py
    ```
    
    This will regenerate the dataset and retrain the model with updated parameters.
    """)
    
    # Display model files info
    if model_exists:
        st.markdown("## 📁 Model Files")
        
        files_info = []
        model_files = ['disease_model.pkl', 'label_encoder.pkl', 'symptoms.pkl']
        
        for file in model_files:
            file_path = os.path.join(models_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                files_info.append({
                    'File': file,
                    'Size (KB)': f"{size / 1024:.2f}",
                    'Status': '✅ Available'
                })
            else:
                files_info.append({
                    'File': file,
                    'Size (KB)': 'N/A',
                    'Status': '❌ Missing'
                })
        
        df = pd.DataFrame(files_info)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()