# 🛡️ Phishing Website Detection System

A complete, production-ready machine learning application designed to detect and flag potentially malicious or phishing URLs. Built with Python, scikit-learn, and Streamlit.

## 🌟 Project Overview

This project provides a web-based dashboard where users can input a website URL and receive an instant prediction on whether the URL is **Safe**, **Suspicious**, or **Phishing**. 

The system works by extracting structural and lexical features from the URL (without actively visiting the potentially dangerous site) and passing these features through a trained Random Forest classification model.

## 🚀 Features

- **Instant URL Analysis**: Predicts the risk level of any given URL.
- **Robust Feature Extraction**: Analyzes URL length, subdomains, special characters, and suspicious keywords.
- **Machine Learning**: Powered by a highly accurate `RandomForestClassifier`.
- **Cyber-Security Themed UI**: A modern, dark-themed Streamlit dashboard with probability charts and technical breakdowns.
- **Zero-Touch Synthetic Data Generation**: Automatically generates a realistic synthetic dataset if a local dataset is not provided, making setup effortless.

## 📊 Dataset Description

The model expects a dataset containing various numerical features derived from URLs, along with a target `label` (0 = Safe, 1 = Suspicious, 2 = Phishing). 

If no dataset is found (`phishing_dataset.csv`), the `train.py` script automatically generates a robust 3000-row synthetic dataset to allow the application to work out-of-the-box. You can replace this with a real dataset from Kaggle or UCI Machine Learning Repository later.

## ⚙️ How Feature Extraction Works

When a URL is submitted, `feature_extractor.py` processes it and extracts the following numerical features:
1. **URL Length**: Extremely long URLs are often used to hide the true domain.
2. **Dot/Dash/Slash/Digit Count**: High counts of special characters often indicate obfuscation.
3. **Subdomain Count**: Multiple subdomains (e.g., `login.secure.bank.com`) trick users into trusting a domain.
4. **Presence of @ / HTTPS / IP**: Use of IP addresses instead of domains or the `@` symbol are strong phishing indicators.
5. **Suspicious Keywords**: Checks for manipulative words like "login", "verify", or "secure".
6. **URL Shorteners**: Checks if the domain belongs to a known URL shortening service (like bit.ly).

## 🛠️ Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/phishing-detector.git
   cd phishing-detector
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 How to Train the Model

Before running the web app, you must train the machine learning model.

Run the following command:
```bash
python train.py
```
*Note: If `phishing_dataset.csv` is not present, this script will automatically generate a synthetic dataset, train the model, and save `model.pkl` and `label_encoder.pkl`.*

## 💻 How to Run Locally

Once the model is trained, start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## ☁️ How to Deploy to Streamlit Community Cloud

1. Push this entire project to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in.
3. Click **"New app"**.
4. Select your repository, branch, and set the **Main file path** to `app.py`.
5. Click **"Deploy!"**. Streamlit will automatically install the packages from `requirements.txt` and launch your app.

## 🔍 Example URLs to Test

**Safe URLs:**
- `https://www.google.com`
- `https://github.com`
- `https://openai.com`

**Suspicious / Phishing URLs:**
- `http://paypal-login-security-update.com`
- `http://verify-bank-account-now.net`
- `http://192.168.1.1/login`

## 📸 Screenshots

![Dashboard Placeholder](https://via.placeholder.com/800x400.png?text=Dashboard+Screenshot+Here)
*(Replace this with an actual screenshot of the app running)*
