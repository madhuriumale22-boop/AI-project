# AI Disease Prediction System

## 🎯 Objective
Create a system that predicts possible diseases based on user-entered symptoms using machine learning.

## 🚀 Features
- **Symptom-based Disease Prediction**: Input symptoms and get disease predictions
- **Machine Learning Model**: Uses Random Forest classifier for accurate predictions
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Data Visualization**: Charts and graphs to visualize prediction results
- **Model Performance Metrics**: Accuracy, precision, recall, and F1-score evaluation

## 🛠️ Technologies Used
- **Python 3.8+**
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn, joblib, streamlit
- **Machine Learning**: Random Forest Classifier
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy

## 📁 Project Structure
```
AI_Disease_Prediction/
├── data/
│   ├── disease_dataset.csv
│   └── processed_data.csv
├── models/
│   ├── disease_model.pkl
│   └── label_encoder.pkl
├── src/
│   ├── data_generator.py
│   ├── model_trainer.py
│   ├── predictor.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Installation & Setup

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate dataset and train model**:
   ```bash
   python src/data_generator.py
   python src/model_trainer.py
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
The system uses a synthetic dataset containing:
- **Symptoms**: 20 common symptoms (fever, cough, headache, etc.)
- **Diseases**: 10 common diseases (flu, cold, migraine, etc.)
- **1000+ records** for training the model

## 🎯 Usage
1. Open the web application
2. Select symptoms from the available options
3. Click "Predict Disease" to get predictions
4. View the predicted disease with confidence score
5. See additional information and recommendations

## 📈 Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~85-90% on test data
- **Features**: Binary encoding of symptoms
- **Cross-validation**: 5-fold validation for robust evaluation

## 🔮 Future Enhancements
- Integration with real medical datasets
- Support for more symptoms and diseases
- User history and tracking
- Doctor recommendations
- Mobile app version

## ⚠️ Disclaimer
This system is for educational purposes only. Always consult with healthcare professionals for medical advice.