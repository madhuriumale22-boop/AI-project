import joblib
import validators
import os

def load_model(model_path="model.pkl"):
    """Loads the trained RandomForestClassifier model."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def load_label_encoder(encoder_path="label_encoder.pkl"):
    """Loads the fitted LabelEncoder."""
    if os.path.exists(encoder_path):
        return joblib.load(encoder_path)
    return None

def validate_url(url):
    """
    Validates if the provided string is a properly formatted URL.
    Prepends 'http://' if the scheme is missing to assist validation.
    """
    if not url:
        return False
        
    url = url.strip()
    if not (url.startswith('http://') or url.startswith('https://')):
        url_to_check = 'http://' + url
    else:
        url_to_check = url
        
    return validators.url(url_to_check)

def get_prediction_label(prediction_index):
    """Maps the numeric prediction to a human-readable label."""
    mapping = {
        0: "Safe",
        1: "Suspicious",
        2: "Phishing"
    }
    return mapping.get(prediction_index, "Unknown")

def get_risk_color(label):
    """Returns the CSS color associated with a risk level."""
    colors = {
        "Safe": "#00FF00",       # Neon Green
        "Suspicious": "#FFA500", # Orange
        "Phishing": "#FF0000",   # Red
        "Unknown": "#808080"     # Gray
    }
    return colors.get(label, "#FFFFFF")

def format_confidence(probabilities):
    """
    Formats the probability array into a readable percentage dictionary.
    Assumes order: [Safe, Suspicious, Phishing] if 3 classes, or [Safe, Phishing] if 2 classes.
    """
    if len(probabilities) == 3:
        return {
            "Safe": f"{probabilities[0] * 100:.2f}%",
            "Suspicious": f"{probabilities[1] * 100:.2f}%",
            "Phishing": f"{probabilities[2] * 100:.2f}%"
        }
    elif len(probabilities) == 2:
        return {
            "Safe": f"{probabilities[0] * 100:.2f}%",
            "Phishing": f"{probabilities[1] * 100:.2f}%"
        }
    return {}
