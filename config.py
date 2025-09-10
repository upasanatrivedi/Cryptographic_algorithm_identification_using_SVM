import os

# ===============================
# General Project Settings
# ===============================
# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
#LOGS_DIR = os.path.join(ROOT_DIR, "logs")
REPORT_DIR = os.path.join(ROOT_DIR, "report")
SVM_CLASSIFIERS_DIR = os.path.join(ROOT_DIR, "svm_classifiers")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# Dataset paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "ciphertexts.csv")
PREPROCESSED_DATA_PATH = os.path.join(DATA_DIR, "preprocessed_ciphers.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "extracted_features.csv")

# ===============================
# Machine Learning Settings
# ===============================
# SVM Hyperparameters
SVM_KERNEL = "rbf"          # Options: 'linear', 'poly', 'rbf', 'sigmoid'
SVM_C = 1.0                 # Regularization parameter
SVM_GAMMA = "scale"         # Kernel coefficient (Options: 'scale', 'auto')
SVM_TOLERANCE = 0.001       # Tolerance for stopping criteria

# Supported cryptographic algorithms
ALGORITHMS = ["AES", "DES", "RSA", "Blowfish"]

# Pairs of algorithms for one-vs-one classification
CLASSIFIER_PAIRS = [
    ("AES", "DES"),
    ("AES", "RSA"),
    ("AES", "Blowfish"),
    ("DES", "RSA"),
    ("DES", "Blowfish"),
    ("RSA", "Blowfish"),
]

MODEL_PATHS = {
    "AES-DES": os.path.join(MODELS_DIR, "svm_classifier_AES_vs_DES.joblib"),
    "AES-BLOWFISH": os.path.join(MODELS_DIR, "svm_classifier_AES_vs_Blowfish.joblib"),
    "AES-RSA": os.path.join(MODELS_DIR, "svm_classifier_AES_vs_RSA.joblib"),
    "BLOWFISH-RSA": os.path.join(MODELS_DIR, "svm_classifier_Blowfish_vs_RSA.joblib"),
    "DES-RSA": os.path.join(MODELS_DIR, "svm_classifier_DES_vs_RSA.joblib"),
    "DES-BLOWFISH": os.path.join(MODELS_DIR, "svm_classifier_DES_vs_Blowfish.joblib"),
}

# Cross-validation settings
CROSS_VALIDATION_FOLDS = 5

# Performance thresholds
MIN_ACCURACY = 0.85
MIN_CONFIDENCE_SCORE = 0.8

# Feature extraction settings
PCA_COMPONENTS = 20         # Number of components for PCA

# Model storage format
MODEL_SAVE_FORMAT = "joblib"  # Options: 'joblib', 'pickle'

# Random seed for reproducibility
RANDOM_SEED = 42

# ===============================
# API and Server Settings
# ===============================
# API settings for FastAPI/Flask
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG_MODE = True

# ===============================
# Testing and Evaluation Settings
# ===============================
# Maximum number of iterations for model training
MAX_ITER = 1000

# Feedback system settings
FEEDBACK_COLLECTION_ENABLED = True
