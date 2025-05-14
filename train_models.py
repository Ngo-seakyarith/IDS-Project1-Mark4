import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
import numpy as np
from scipy.sparse import csr_matrix

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
DATA_FILE = 'com-full-backup.csv'
TEXT_COLUMN = 'Symptoms'
LABEL_COLUMN = 'Class'
TEST_SIZE = 0.05
RANDOM_STATE = 42
MODELS_DIR = 'trained_models'

# --- Load and Prepare Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Handle missing values
print("Checking for missing values...")
df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
print(f"Data shape after dropping missing values: {df.shape}")

# Ensure columns are strings
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

# Encode labels
print("Encoding labels...")
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df[LABEL_COLUMN])
print(f"Number of unique classes: {len(label_encoder.classes_)}")
# Save the label encoder and classes
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(label_encoder, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
print(f"Label encoder saved to {os.path.join(MODELS_DIR, 'label_encoder.joblib')}")

# --- Split Data ---
print(f"Splitting data (Test size: {TEST_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN],
    df['encoded_labels'],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['encoded_labels'] # Stratify to maintain class distribution
)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# --- TF-IDF Vectorization ---
print("Applying TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000) # Limit features for efficiency
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF matrix shape (Train): {X_train_tfidf.shape}")
# Save the vectorizer
joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
print(f"TF-IDF vectorizer saved to {os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')}")

# --- Model Training and Evaluation ---
base_models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC(random_state=RANDOM_STATE, tol=1e-5, max_iter=2000),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=300, 
        random_state=RANDOM_STATE, 
        early_stopping=True
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=1000, 
        random_state=RANDOM_STATE, 
        tol=1e-3
    ),
    "SGD": SGDClassifier(
        max_iter=1000, 
        tol=1e-3, 
        random_state=RANDOM_STATE,
        loss='modified_huber'
    )
}

# Train models
print("\n--- Training Models ---")
results = {}

for name, model in base_models.items():
    print(f"Training {name}...")
    try:
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} - Test Accuracy: {accuracy:.4f}")

        # Save model with compression
        model_filename = os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_")}_model.joblib')
        joblib.dump(model, model_filename, compress=3)
        print(f"{name} model saved to {model_filename}")

    except Exception as e:
        print(f"Error training/evaluating {name}: {e}")
        results[name] = None

print("\n--- Training Summary ---")
for name, accuracy in results.items():
    if accuracy is not None:
        print(f"{name}: {accuracy:.4f}")
    else:
        print(f"{name}: Training failed")

print(f"\nAll models and vectorizer saved in '{MODELS_DIR}' directory.")
