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
st.set_page_config(layout="wide", initial_sidebar_state="expanded") # Keep sidebar open

# --- Theme Management ---
# Initialize session state for theme if it doesn't exist
if 'theme' not in st.session_state:
    st.session_state.theme = 'light' # Default to light theme
    # Initialize the toggle state key if it doesn't exist, mirroring the theme
    st.session_state.theme_toggle_main = (st.session_state.theme == 'dark')

# Theme state is now managed by the toggle in the main area

# --- Define CSS Styles ---
light_theme_css = """
<style>
    /* --- Light Theme --- */
    body { background-color: #f0f2f6; color: #1f2937; }
    [data-testid="stAppViewContainer"] { background: #f0f2f6; }
    [data-testid="stSidebar"] { background-color: #ffffff; }

    .custom-card, .prediction-box, .model-details, .streamlit-expanderHeader, .stTextArea textarea {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #e5e7eb; /* Subtle border */
    }
    .custom-card { padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin: 10px 0; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    .model-details { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .streamlit-expanderHeader { border-radius: 8px; margin-bottom: 5px; }
    .stTextArea textarea { border-radius: 8px; }

    h1, h2, h3, h4, h5, h6 { color: #1e3a8a !important; font-weight: 600 !important; margin-bottom: 0.5em !important; }
    p, li, span, .prediction-text, .streamlit-expanderHeader { color: #1f2937 !important; }
    .score-text, [data-testid="stMetricLabel"], .caption-text { color: #4b5563 !important; }
    .prediction-text { font-size: 1.1rem; font-weight: 500; }
    .score-text { font-size: 1rem; }
    .caption-text { font-size: 0.875rem; }

    [data-testid="stMetricValue"] { color: #1e3a8a !important; font-weight: 600 !important; }

    .stButton button {
        background-color: #1e3a8a; color: #ffffff; font-weight: 600;
        padding: 0.5rem 2rem; border-radius: 8px; border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton button:hover { background-color: #1e40af; }
    .stButton button:active { background-color: #1d4ed8; } /* Added active state */

    /* Specific prediction box border color handled inline */

    /* Ensure toggle visibility in light mode - Simpler Approach */
    .toggle-container div[data-baseweb="toggle"] { /* Target the toggle base element */
        border: 1px solid #adb5bd !important; /* Add a visible gray border */
        background-color: #f8f9fa !important; /* Slightly off-white background */
    }
    .toggle-container label { /* Ensure label (emoji) is dark */
         color: #212529 !important;
         margin-bottom: 0 !important; /* Prevent extra spacing */
         display: flex !important;
         align-items: center !important;
    }
    /* Optional: Style the knob if needed, might require browser inspection */
    /* .toggle-container div[data-baseweb="toggle-knob"] { background-color: #ced4da !important; } */
</style>
"""

dark_theme_css = """
<style>
    /* --- Dark Theme --- */
    body { background-color: #0f172a; color: #e2e8f0; } /* Dark blue-gray */
    [data-testid="stAppViewContainer"] { background: #0f172a; }
    [data-testid="stSidebar"] { background-color: #1e293b; } /* Slightly lighter sidebar */

    .custom-card, .prediction-box, .model-details, .streamlit-expanderHeader, .stTextArea textarea {
        background-color: #1e293b; /* Dark slate */
        color: #e2e8f0; /* Light gray text */
        border: 1px solid #334155; /* Subtle border */
    }
    .custom-card { padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); margin: 10px 0; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); }
    .model-details { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .streamlit-expanderHeader { border-radius: 8px; margin-bottom: 5px; }
    .stTextArea textarea { border-radius: 8px; }

    h1, h2, h3, h4, h5, h6 { color: #93c5fd !important; font-weight: 600 !important; margin-bottom: 0.5em !important; } /* Light blue */
    p, li, span, .prediction-text, .streamlit-expanderHeader { color: #e2e8f0 !important; } /* Light gray */
    .score-text, [data-testid="stMetricLabel"], .caption-text { color: #94a3b8 !important; } /* Medium gray */
    .prediction-text { font-size: 1.1rem; font-weight: 500; }
    .score-text { font-size: 1rem; }
    .caption-text { font-size: 0.875rem; }

    [data-testid="stMetricValue"] { color: #93c5fd !important; font-weight: 600 !important; } /* Light blue */

    .stButton button {
        background-color: #3b82f6; /* Brighter blue */
        color: #ffffff; font-weight: 600;
        padding: 0.5rem 2rem; border-radius: 8px; border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stButton button:hover { background-color: #60a5fa; }
    .stButton button:active { background-color: #2563eb; } /* Added active state */

    /* Specific prediction box border color handled inline */

    /* Ensure toggle visibility */
    [data-testid="stToggle"] label {
        color: #e2e8f0 !important; /* Light text/emoji for dark mode */
        display: flex; /* Align icon better */
        align-items: center;
    }
</style>
"""

# Apply the selected theme's CSS
st.markdown(dark_theme_css if st.session_state.theme == 'dark' else light_theme_css, unsafe_allow_html=True)

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
# Sidebar is kept clean, controls are in the main area or implicit

# Callback function to update theme based on toggle state
def update_theme():
    # Read the current state of the toggle widget directly from session_state
    is_dark = st.session_state.theme_toggle_main
    st.session_state.theme = 'dark' if is_dark else 'light'

# Main title and Theme Toggle
col_title, col_toggle = st.columns([0.85, 0.15]) # Adjust ratio as needed
with col_title:
    st.title("🏥 AI-Powered Disease Prediction")
with col_toggle:
    st.markdown("<div class='toggle-container' style='margin-top: 25px;'>", unsafe_allow_html=True)
    # Use the on_change callback
    st.toggle(
        "🌙",
        key="theme_toggle_main", # State is stored here
        on_change=update_theme,  # Function to call when toggled
        help="Toggle Dark Mode"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    # Theme is now updated via the callback, no direct update here needed.

st.markdown("""
<div class='custom-card'>
    <h4>Advanced Multi-Model Disease Prediction System</h4>
    <p>This system utilizes multiple AI models to analyze symptoms and provide potential disease predictions. Use the toggle (🌙) above to switch themes.</p>
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
                        # Adjusted colors for better visibility in both themes
                        "#16a34a" if confidence >= 75 else  # Green 600
                        "#f59e0b" if confidence >= 50 else  # Amber 500
                        "#ef4444"  # Red 500
                    )
                    border_color_dark = ( # Slightly lighter for dark mode contrast
                        "#22c55e" if confidence >= 75 else  # Green 500
                        "#facc15" if confidence >= 50 else  # Yellow 400
                        "#f87171"  # Red 400
                    )
                    final_border_color = border_color_dark if st.session_state.theme == 'dark' else confidence_color

                    # Determine text color based on confidence for better contrast inside the box
                    text_color_light = "#1f2937" # Default dark text for light theme
                    text_color_dark = "#e2e8f0" # Default light text for dark theme
                    final_text_color = text_color_dark if st.session_state.theme == 'dark' else text_color_light
                    final_score_color = "#94a3b8" if st.session_state.theme == 'dark' else "#4b5563" # Muted score color

                    st.markdown(f"""
                    <div class='prediction-box' style='border-left: 5px solid {final_border_color};'>
                        <div class='prediction-text' style='color: {final_text_color} !important;'>
                            {["🥇", "🥈", "🥉"][i]} <strong>{disease}</strong>
                        </div>
                        <div class='score-text' style='color: {final_score_color} !important;'>
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
                                    <div class='prediction-text'><strong>{cls}</strong></div>
                                    <div class='score-text'>{conf_pct:.1f}% Confidence</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class='caption-text'>
                                    &nbsp;&nbsp;• {cls} ({conf_pct:.1f}%)
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
