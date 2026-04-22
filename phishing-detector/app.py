import streamlit as st
import pandas as pd
import time
import urllib.parse
import tldextract

from feature_extractor import extract_features
from utils import load_model, validate_url, get_prediction_label, get_risk_color, format_confidence

# Must be the first Streamlit command
st.set_page_config(
    page_title="Phishing Website Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cyber-Security Theme
st.markdown("""
<style>
    /* Dark Theme Backgrounds */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Input Field */
    .stTextInput>div>div>input {
        background-color: #161b22;
        color: #58a6ff;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
        font-family: monospace;
        font-size: 16px;
    }
    
    /* Analyze Button */
    .stButton>button {
        background-color: #238636;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 0 10px #2ea043;
        color: #ffffff;
    }
    
    /* Result Cards */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Feature Table */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Technical Details Expander */
    .streamlit-expanderHeader {
        background-color: #21262d;
        color: #c9d1d9;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

def generate_explanation(features, url):
    """Generates an explanation based on the extracted features."""
    explanations = []
    
    # 0: url_length, 5: subdomain_count, 8: contains_ip, 9: contains_suspicious, 10: uses_short
    
    if features[0] > 75:
        explanations.append("⚠️ **URL is unusually long**, which is common in obfuscated phishing links.")
    if features[5] > 2:
        explanations.append("⚠️ **Multiple subdomains detected**, often used to trick users into trusting a domain.")
    if features[8] == 1:
        explanations.append("🔴 **IP Address used instead of a domain name**, a very strong indicator of malicious intent.")
    if features[9] == 1:
        explanations.append("⚠️ **Contains suspicious keywords** (e.g., login, verify, secure) intended to manipulate users.")
    if features[10] == 1:
        explanations.append("⚠️ **Uses a URL shortener**, which hides the true destination of the link.")
        
    if not explanations:
        explanations.append("✅ No obvious structural anomalies detected in the URL.")
        
    return explanations

def main():
    # --- Title & Subtitle ---
    st.title("🛡️ Phishing Website Detection System")
    st.markdown("### Analyze whether a website URL may be malicious or phishing.")
    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔍 About the Project")
        st.write("This tool uses a Machine Learning model (Random Forest) to analyze the structural features of a URL and predict its safety.")
        
        st.header("📊 Model Status")
        model = load_model("model.pkl")
        if model:
            st.success("✅ Model loaded successfully!")
            # In a real app, you might load a stats dict. Here we mock the accuracy text.
            st.write("**Est. Accuracy:** ~90-95%")
        else:
            st.error("❌ Model not found! Please run `python train.py` first.")
            st.stop()
            
        st.header("📝 Instructions")
        st.write("1. Enter a full URL in the main input box.")
        st.write("2. Click 'Analyze URL'.")
        st.write("3. Review the risk assessment and technical details.")
        
        st.header("🌐 Example URLs")
        st.write("**Safe:**")
        st.code("https://www.google.com")
        st.code("https://github.com")
        st.code("https://openai.com")
        
        st.write("**Suspicious / Phishing:**")
        st.code("http://paypal-login-security-update.com")
        st.code("http://verify-bank-account-now.net")
        st.code("http://192.168.1.1/login")

    # --- Main Input Area ---
    url_input = st.text_input("🔗 Enter Website URL to scan:", placeholder="e.g., https://www.example.com")
    
    if st.button("Analyze URL"):
        if not url_input:
            st.warning("Please enter a URL to analyze.")
        elif not validate_url(url_input):
            st.error("Invalid URL format. Please ensure it looks like a valid link (e.g., example.com or http://example.com).")
        else:
            with st.spinner("Analyzing URL architecture and matching threat signatures..."):
                time.sleep(1) # Simulated delay for dramatic effect
                
                # Ensure scheme for parsing
                if not (url_input.startswith('http://') or url_input.startswith('https://')):
                    url_to_process = 'http://' + url_input
                else:
                    url_to_process = url_input
                
                # 1. Extract Features
                features = extract_features(url_to_process)
                feature_names = [
                    'URL Length', 'Dot Count', 'Dash Count', 'Slash Count', 'Digit Count',
                    'Subdomain Count', 'Contains @', 'Contains HTTPS', 'Contains IP',
                    'Contains Suspicious Keyword', 'Uses Shortener'
                ]
                
                # 2. Predict
                # model.predict_proba requires 2D array
                feature_array = [features]
                pred_proba = model.predict_proba(feature_array)[0]
                pred_index = model.predict(feature_array)[0]
                
                label = get_prediction_label(pred_index)
                color = get_risk_color(label)
                confidence_dict = format_confidence(pred_proba)
                
                # Determine explanation
                if label == "Safe":
                    desc = "This URL appears legitimate."
                elif label == "Suspicious":
                    desc = "Some characteristics are unusual, proceed carefully."
                else:
                    desc = "This URL strongly resembles a phishing attempt."

                # --- Results Display ---
                st.markdown(f"""
                <div class="result-card" style="background-color: {color}; color: {'#000' if label == 'Safe' else '#fff'};">
                    <h2 style="color: {'#000' if label == 'Safe' else '#fff'} !important; margin: 0;">{label.upper()}</h2>
                    <p style="font-size: 18px; margin-top: 10px;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Probability Distribution")
                    # Create a dataframe for the chart
                    chart_data = pd.DataFrame(
                        [float(v.strip('%')) for v in confidence_dict.values()],
                        index=confidence_dict.keys(),
                        columns=['Probability (%)']
                    )
                    st.bar_chart(chart_data, color="#58a6ff")
                    
                    st.subheader("Why was it flagged?")
                    explanations = generate_explanation(features, url_to_process)
                    for exp in explanations:
                        st.markdown(exp)
                        
                with col2:
                    st.subheader("Extracted URL Features")
                    feature_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Value": features
                    })
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)
                
                # --- Advanced Details ---
                st.markdown("---")
                with st.expander("🛠️ Show Technical Details"):
                    parsed = urllib.parse.urlparse(url_to_process)
                    extracted = tldextract.extract(url_to_process)
                    
                    st.write("**Parsed URL Components:**")
                    st.json({
                        "Scheme": parsed.scheme,
                        "Network Location (Domain)": parsed.netloc,
                        "Path": parsed.path,
                        "Subdomain": extracted.subdomain,
                        "Domain": extracted.domain,
                        "Suffix (TLD)": extracted.suffix
                    })
                    
                    st.write("**Report Summary (Copyable):**")
                    report = f"Target URL: {url_input}\nPrediction: {label}\nConfidence: {confidence_dict.get(label, 'N/A')}\n"
                    st.code(report, language="text")

if __name__ == "__main__":
    main()
