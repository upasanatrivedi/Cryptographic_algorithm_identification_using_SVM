import random
from joblib import load
import pandas as pd
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import MODEL_PATHS
from feature_extraction import calculate_char_frequency, calculate_entropy, calculate_length

# Load model paths from config
model_paths = MODEL_PATHS

def preprocess_input(input_ciphertext):
    """ Preprocess the input ciphertext and extract features. """
    len_cipher = calculate_length(input_ciphertext)
    en_cipher = calculate_entropy(input_ciphertext)

    # Prepare feature set
    data = {
        'Input': input_ciphertext,
        'length': len_cipher,
        'entropy': en_cipher,
    }

    # Create DataFrame with the input data
    features = pd.DataFrame([data])

    # Generate feature columns for character frequencies
    char_freq_columns = [f"freq_{char}" for char in "0123456789ABCDEF"]
    char_freq_values = calculate_char_frequency(input_ciphertext)

    # Add character frequencies to DataFrame
    char_freq_df = pd.DataFrame([char_freq_values], columns=char_freq_columns)
    features = pd.concat([features, char_freq_df], axis=1)
    features = features.drop(columns=["Input"])

    return features


def classify_with_decision_function(features, model_paths):
    """Run all classifiers and return predictions with decision function scores."""
    def run_classifiers(features, model_paths):
        predictions = {}
        decision_scores = {}

        for classifier_name, model_path in model_paths.items():
            try:
                # Load the model from the given path
                svm_classifier = load(model_path)

                # Get prediction and decision function scores
                prediction = svm_classifier.predict(features)[0]  # Single instance prediction
                decision_function = svm_classifier.decision_function(features)[0]  # Scalar or array

                # If decision function is scalar (binary SVM), take absolute value
                if isinstance(decision_function, float):
                    decision_scores[classifier_name] = abs(decision_function)
                else:
                    decision_scores[classifier_name] = max(abs(decision_function))

                predictions[classifier_name] = prediction

            except Exception as e:
                print(f"Error loading classifier {classifier_name} from {model_path}: {e}")

        return predictions, decision_scores

    def majority_vote(predictions):
        """Determine the majority vote among all predictions."""
        vote_count = {}
        for prediction in predictions.values():
            if prediction in vote_count:
                vote_count[prediction] += 1
            else:
                vote_count[prediction] = 1
        return max(vote_count, key=vote_count.get), vote_count

    # Step 1: Run all classifiers
    predictions, decision_scores = run_classifiers(features, model_paths)
    print(f"Predictions: {predictions}")
    print(f"Decision Scores: {decision_scores}")

    # Step 2: Apply majority voting
    final_prediction, vote_count = majority_vote(predictions)
    print(f"Vote count: {vote_count}")

    # Step 3: Resolve ties by confidence scores
    if list(vote_count.values()).count(max(vote_count.values())) > 1:
        # If there is a tie, use confidence scores to break it
        tied_predictions = [key for key, val in vote_count.items() if val == max(vote_count.values())]
        print(f"Tied predictions: {tied_predictions}")

        # Choose the prediction with the highest confidence score
        best_prediction = None
        highest_score = -float('inf')

        for classifier_name, prediction in predictions.items():
            if prediction in tied_predictions and decision_scores[classifier_name] > highest_score:
                highest_score = decision_scores[classifier_name]
                best_prediction = prediction

        return best_prediction

    return final_prediction


# This function will be used in app.py to perform the actual classification
def classify_ciphertext(input_ciphertext):
    """Function to classify ciphertext."""
    features = preprocess_input(input_ciphertext)
    final_prediction = classify_with_decision_function(features, model_paths)
    return final_prediction
