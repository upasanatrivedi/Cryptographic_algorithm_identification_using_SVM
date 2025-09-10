import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import optuna
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import FEATURES_PATH, MODELS_DIR, RANDOM_SEED


def transform_features(features):
    """
    Transforms the extracted_features.csv into a labeled dataset with unified columns for DES and RSA.
    """
    # Extract features for DES
    DES_features = features[[col for col in features.columns if "_DES" in col]]
    DES_features.columns = [col.replace("_DES", "") for col in DES_features.columns]  # Standardize column names
    DES_features["Label"] = "DES"  # Add Label column for DES

    # Extract features for RSA
    RSA_features = features[[col for col in features.columns if "_RSA" in col]]
    RSA_features.columns = [col.replace("_RSA", "") for col in RSA_features.columns]  # Standardize column names
    RSA_features["Label"] = "RSA"  # Add Label column for RSA

    # Combine DES and RSA features into a single dataframe
    transformed_features = pd.concat([DES_features, RSA_features], ignore_index=True)

    return transformed_features


def objective(trial, X_train, y_train, X_test, y_test):
    """
    Objective function for Optuna to optimize SVM hyperparameters.
    """
    # Define the hyperparameter search space
    C = trial.suggest_loguniform("C", 0.1, 100)  # Log-scale for C
    gamma = trial.suggest_loguniform("gamma", 0.001, 1)  # Log-scale for gamma
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])  # Choice of kernel

    # Train an SVM with the suggested hyperparameters
    svm = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def svm_classifier_DES_vs_RSA(features):
    """
    Trains and evaluates an SVM classifier for DES vs RSA classification with Bayesian optimization using Optuna.
    Saves the trained model to the models/ directory.
    """
    # Separate features (X) and labels (y)
    X = features.drop(columns=["Label"])
    y = features["Label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Optimize hyperparameters using Optuna
    study = optuna.create_study(direction="maximize")  # Maximize accuracy
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # Train the final model with the best hyperparameters
    best_model = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel=best_params["kernel"],
        probability=True,
        random_state=RANDOM_SEED
    )
    best_model.fit(X_train, y_train)

    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of DES vs RSA classifier: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model
    model_path = os.path.join(MODELS_DIR, "svm_classifier_DES_vs_RSA.joblib")
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")


def main():
    """
    Main function to execute feature transformation and train the DES vs RSA SVM classifier.
    """
    # Load features
    print("Loading features from extracted_features.csv...")
    features = pd.read_csv(FEATURES_PATH)

    # Transform features to include Label column and unify structure
    print("Transforming features...")
    transformed_features = transform_features(features)

    # Train and evaluate the DES vs RSA SVM classifier
    print("Training DES vs RSA SVM classifier...")
    svm_classifier_DES_vs_RSA(transformed_features)


if __name__ == "__main__":
    print("Starting DES vs RSA classification...")
    main()
