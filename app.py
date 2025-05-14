import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from collections import Counter
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Streamlit Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide")

# Add custom CSS for modern design
st.markdown("""
<style>
    /* Base theme */
    [data-testid="stAppViewContainer"] {
        background: #f0f2f6;
    }

    /* Card styling */
    .custom-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }

    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a8a !important;
        font-weight: 600 !important;
        margin-bottom: 0.5em !important;
    }

    p, li, span {
        color: #1f2937 !important;
    }

    /* Prediction box */
    .prediction-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Prediction text */
    .prediction-text {
        color: #1f2937 !important;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* Score text */
    .score-text {
        color: #4b5563 !important;
        font-size: 1rem;
    }

    /* Model details */
    .model-details {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }

    /* Button styling */
    .stButton button {
        background-color: #1e3a8a;
        color: #ffffff;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton button:hover {
        background-color: #1e40af;
    }

    /* Input area styling */
    .stTextArea textarea {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }

    /* Caption and helper text */
    .caption-text {
        color: #6b7280 !important;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
MODELS_DIR = 'trained_models'
VECTORIZER_FILE = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, 'label_encoder.joblib')
MODEL_FILES = {
    "Naive Bayes": os.path.join(MODELS_DIR, 'naive_bayes_model.joblib'),
    "SVM": os.path.join(MODELS_DIR, 'svm_model.joblib'),
    "Neural Network": os.path.join(MODELS_DIR, 'neural_network_model.joblib'),
    "KNN": os.path.join(MODELS_DIR, 'knn_model.joblib'),
    "Passive Aggressive": os.path.join(MODELS_DIR, 'passive_aggressive_model.joblib'),
    "SGD": os.path.join(MODELS_DIR, 'sgd_model.joblib')
}

# --- Load Models and Preprocessing Objects ---
# Use st.cache_resource to load models only once
@st.cache_resource
def load_resources():
    """Loads the vectorizer, label encoder, and all models."""
    resources = {}
    try:
        resources['vectorizer'] = joblib.load(VECTORIZER_FILE)
        resources['label_encoder'] = joblib.load(LABEL_ENCODER_FILE)
        resources['models'] = {}
        for name, path in MODEL_FILES.items():
            if os.path.exists(path):
                resources['models'][name] = joblib.load(path)
            else:
                st.error(f"Model file not found: {path}")
                resources['models'][name] = None
        resources['models'] = {k: v for k, v in resources['models'].items() if v is not None}
        if not resources['models']:
            st.error("No models were loaded successfully. Please ensure models are trained and present in the 'trained_models' directory.")
            return None
        return resources
    except FileNotFoundError as e:
        st.error(f"Error loading resource: {e}. Please ensure 'train_models.py' has been run successfully.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return None

resources = load_resources()

# --- Streamlit App Interface ---
# st.set_page_config(layout="wide") # Moved to the top
st.title("🏥 AI-Powered Disease Prediction")
st.markdown("""
<div class='custom-card'>
    <h3>Advanced Multi-Model Disease Prediction System</h3>
    <p>This sophisticated system utilizes multiple AI models to analyze symptoms and provide accurate disease predictions.</p>
</div>
""", unsafe_allow_html=True)

# Improved input section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📝 Symptom Analysis")
symptoms_input = st.text_area(
    "Enter Patient Symptoms:",
    height=120,
    placeholder="Please describe the symptoms in detail (e.g., high fever for 3 days, persistent dry cough, fatigue)",
    help="For best results, provide detailed symptoms separated by commas"
)

# Add a professional-looking analyze button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_button = st.button("🔍 Analyze Symptoms", use_container_width=True)

# Add this helper function at the top level of your code
def get_prediction_probabilities(model, input_data):
    """Get probability-like scores for all models."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(input_data)[0]
    elif hasattr(model, 'decision_function'):
        # Convert decision function scores to pseudo-probabilities
        decision_scores = model.decision_function(input_data)
        if decision_scores.ndim == 1:
            # Binary classification
            scores = np.array([[-s, s] for s in decision_scores])[0]
        else:
            # Multi-class
            scores = decision_scores[0]
        # Apply softmax to convert scores to pseudo-probabilities
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
    else:
        # Fallback for models with neither method
        pred = model.predict(input_data)[0]
        # Create a one-hot like probability distribution
        proba = np.zeros(len(resources['label_encoder'].classes_))
        proba[pred] = 1
        return proba

if analyze_button:
    if not symptoms_input:
        st.warning("Please enter symptoms first.")
    elif not resources:
        st.error("Models could not be loaded.")
    else:
        try:
            input_vector = resources['vectorizer'].transform([symptoms_input])
            weighted_predictions = {}
            model_predictions = {}

            for name, model in resources['models'].items():
                try:
                    # Handle all models the same way now
                    pred = model.predict(input_vector)[0]
                    proba = get_prediction_probabilities(model, input_vector)

                    # Get prediction class
                    pred_class = resources['label_encoder'].inverse_transform([pred])[0]

                    # Get confidence scores
                    confidence = proba.max()
                    top_3_indices = np.argsort(proba)[-3:][::-1]
                    top_3_classes = resources['label_encoder'].inverse_transform(top_3_indices)
                    top_3_probs = proba[top_3_indices]

                    # Store predictions and confidence scores
                    model_predictions[name] = {
                        'classes': list(top_3_classes),
                        'confidences': list(top_3_probs),
                        'top_prediction': pred_class,
                        'top_confidence': confidence
                    }

                    # Aggregate weighted predictions
                    for cls, conf in zip(top_3_classes, top_3_probs):
                        if cls not in weighted_predictions:
                            weighted_predictions[cls] = 0
                        weighted_predictions[cls] += conf

                except Exception as e:
                    st.error(f"Error with {name} model: {e}")
                    continue  # Skip to next model if there's an error

            # Calculate overall top predictions
            total_weights = sum(weighted_predictions.values())
            normalized_predictions = {
                k: (v / total_weights) * 100
                for k, v in weighted_predictions.items()
            }
            top_3_overall = sorted(
                normalized_predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            # Display Results
            st.markdown("<br>", unsafe_allow_html=True)

            # Main predictions container
            st.markdown("""
            <div class='custom-card'>
                <h2>🎯 Diagnostic Analysis</h2>
            </div>
            """, unsafe_allow_html=True)

            # Create modern layout for results
            col_pred, col_detail = st.columns([6, 4])

            with col_pred:
                for i, (disease, confidence) in enumerate(top_3_overall):
                    confidence_color = (
                        "#15803d" if confidence >= 75 else  # Darker green
                        "#b45309" if confidence >= 50 else  # Darker amber
                        "#991b1b"  # Darker red
                    )

                    st.markdown(f"""
                    <div class='prediction-box' style='border-left: 5px solid {confidence_color};'>
                        <div class='prediction-text' style='color: {confidence_color} !important;'>
                            {["🥇", "🥈", "🥉"][i]} {disease}
                        </div>
                        <div class='score-text'>
                            Confidence Score: {confidence:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_detail:
                st.markdown("""
                <div class='custom-card'>
                    <h3>🔍 Model Analysis</h3>
                </div>
                """, unsafe_allow_html=True)

                for name, preds in model_predictions.items():
                    with st.expander(f"🤖 {name}"):
                        for i, (cls, conf) in enumerate(zip(
                            preds['classes'],
                            preds['confidences']
                        )):
                            conf_pct = conf * 100
                            if i == 0:
                                st.markdown(f"""
                                <div class='model-details'>
                                    <div class='prediction-text'>{cls}</div>
                                    <div class='score-text'>{conf_pct:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='caption-text'>
                                    {cls} ({conf_pct:.1f}%)
                                </div>
                                """, unsafe_allow_html=True)

            # Enhanced summary metrics
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='custom-card'>
                <h3>📊 Analysis Summary</h3>
            </div>
            """, unsafe_allow_html=True)

            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric(
                    "Primary Diagnosis",
                    top_3_overall[0][0],
                    f"{top_3_overall[0][1]:.1f}% Confidence"
                )
            with metric_cols[1]:
                agreement = sum(1 for p in model_predictions.values()
                              if p['top_prediction'] == top_3_overall[0][0])
                st.metric(
                    "Model Consensus",
                    f"{agreement}/{len(model_predictions)}",
                    "Models in Agreement"
                )
            with metric_cols[2]:
                confidence_level = (
                    "High" if top_3_overall[0][1] >= 75 else
                    "Moderate" if top_3_overall[0][1] >= 50 else
                    "Low"
                )
                st.metric(
                    "Confidence Level",
                    confidence_level,
                    "Based on Analysis"
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Add disclaimer at the bottom
st.markdown("---")
st.caption(
    "⚠️ This tool is for educational purposes only. "
    "Always consult with healthcare professionals for medical advice."
)

if __name__ == "__main__":
    import sys
    import subprocess
    import os

    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if running directly (not through streamlit)
    if len(sys.argv) == 1 and not os.environ.get('STREAMLIT_RUN_APP'):  # No command line arguments means direct Python execution
        print("Starting Streamlit app...")
        # Set an environment variable to prevent recursive launching
        os.environ['STREAMLIT_RUN_APP'] = '1'
        subprocess.run(["py", "-m", "streamlit", "run", os.path.join(script_dir, "app.py")], shell=True)
