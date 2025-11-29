import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    # Use absolute path to avoid any relative path issues
    model_path = Path(__file__).parent.parent / "Model" / "credit_card_fraud_model.joblib"
    model_path = str(model_path.absolute())
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)

model = load_model()

# Title and description
st.title("💳 Credit Card Fraud Detection")
st.write("""
This application uses a machine learning model to detect potential credit card fraud.
Enter the transaction details below to check if it's potentially fraudulent.
""")

# Create input fields for features
st.sidebar.header("Transaction Details")

def get_user_input():
    # Create input fields for the most important features
    # Note: In a real app, you'd want to include all features
    amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    v14 = st.sidebar.number_input("V14", value=0.0, step=0.01)
    v4 = st.sidebar.number_input("V4", value=0.0, step=0.01)
    v10 = st.sidebar.number_input("V10", value=0.0, step=0.01)
    
    # Create a dictionary of features
    data = {
        'Time': 0,  # Placeholder, as it's required but not used in prediction
        'V1': 0.0, 'V2': 0.0, 'V3': 0.0, 'V4': v4, 'V5': 0.0,
        'V6': 0.0, 'V7': 0.0, 'V8': 0.0, 'V9': 0.0, 'V10': v10,
        'V11': 0.0, 'V12': 0.0, 'V13': 0.0, 'V14': v14, 'V15': 0.0,
        'V16': 0.0, 'V17': 0.0, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
        'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0,
        'V26': 0.0, 'V27': 0.0, 'V28': 0.0,
        'Amount': amount
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = get_user_input()

# Display the input parameters
st.subheader('Transaction Parameters')
st.write(input_df)

# Make prediction
if st.sidebar.button('Check for Fraud'):
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display results
    st.subheader('Prediction')
    fraud_probability = prediction_proba[0][1] * 100
    
    if prediction[0] == 1:
        st.error(f"🚨 Potential Fraud Detected! ({fraud_probability:.2f}% probability)")
    else:
        st.success(f"✅ Transaction Appears Legitimate ({100 - fraud_probability:.2f}% probability)")
    
    # Show probability
    st.subheader('Prediction Probability')
    st.write(f"Probability of being fraudulent: {fraud_probability:.2f}%")
    
    # Show feature importance (if available)
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        st.subheader('Feature Importance')
        importances = model.named_steps['classifier'].feature_importances_
        features = input_df.columns
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))

# Add some information about the model
st.sidebar.info("""
### About
This model was trained on the [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.
- **Accuracy**: 99.95%
- **Precision**: 93.75%
- **Recall**: 76.53%
""")

# Add a footer
st.markdown("---")
st.markdown("### ℹ️ Note")
st.write("""
- This is a demonstration application for educational purposes only.
- The model's predictions should not be the sole factor in determining transaction legitimacy.
- Always verify suspicious transactions through additional verification methods.
""")
