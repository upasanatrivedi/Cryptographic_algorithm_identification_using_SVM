import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from joblib import dump
import os
import time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import FEATURES_PATH, MODELS_DIR, RANDOM_SEED

def transform_features(features):
    Blowfish_features = features[[col for col in features.columns if "_Blowfish" in col]].copy()
    Blowfish_features.columns = [col.replace("_Blowfish", "") for col in Blowfish_features.columns]
    Blowfish_features["Label"] = "Blowfish"

    DES_features = features[[col for col in features.columns if "_DES" in col]].copy()
    DES_features.columns = [col.replace("_DES", "") for col in DES_features.columns]
    DES_features["Label"] = "DES"

    return pd.concat([Blowfish_features, DES_features], ignore_index=True)

def svm_classifier_Blowfish_vs_DES(features):
    X = features.drop(columns=["Label"])
    y = features["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(C=0.3, gamma='scale', kernel='rbf', probability=True, random_state=RANDOM_SEED)

    start_time = time.time()
    svm.fit(X_train_pca, y_train)
    end_time = time.time()

    training_time = end_time - start_time

    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score((y_test == 'Blowfish').astype(int), svm.predict_proba(X_test_pca)[:, 0])

    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Accuracy of Blowfish vs DES classifier: {accuracy * 100:.2f}%")
    print(f"AUC-ROC Score: {auc_roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(MODELS_DIR, "svm_classifier_Blowfish_v_DES_optimized.joblib")
    dump(svm, model_path)
    print(f"Model saved to {model_path}")

def main():
    features = pd.read_csv(FEATURES_PATH)
    transformed_features = transform_features(features)
    svm_classifier_Blowfish_vs_DES(transformed_features)

if __name__ == "__main__":
    main()
