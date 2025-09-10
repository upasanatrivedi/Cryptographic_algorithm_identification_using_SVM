import pandas as pd
import numpy as np
import os
from config import PREPROCESSED_DATA_PATH, FEATURES_PATH, ALGORITHMS

# Feature extraction functions
def calculate_length(ciphertext):
    """Calculates the length of the ciphertext."""
    return len(ciphertext)

def calculate_entropy(ciphertext):
    """Calculates the entropy of the ciphertext."""
    probabilities = np.array([ciphertext.count(char) / len(ciphertext) for char in set(ciphertext)])
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_char_frequency(ciphertext):
    """Calculates the frequency of each hexadecimal character in the ciphertext."""
    freq = {hex_char: ciphertext.count(hex_char) for hex_char in "0123456789ABCDEF"}
    return [freq[char] / len(ciphertext) for char in "0123456789ABCDEF"]

def extract_features(data):
    """Extracts features from the ciphertexts for all algorithms."""
    features = pd.DataFrame()  # Initialize an empty DataFrame to store all features

    # Loop through each algorithm and extract features
    for algo in ALGORITHMS:
        # Ciphertext length
        features[f"length_{algo}"] = data[algo].apply(calculate_length)

        # Entropy
        features[f"entropy_{algo}"] = data[algo].apply(calculate_entropy)

        # Character frequencies
        char_freq_columns = [f"freq_{char}_{algo}" for char in "0123456789ABCDEF"]
        char_freq_values = data[algo].apply(calculate_char_frequency).to_list()
        char_freq_df = pd.DataFrame(char_freq_values, columns=char_freq_columns)

        # Add character frequencies to the features DataFrame
        features = pd.concat([features, char_freq_df], axis=1)

    return features

def save_features(features, file_path):
    """Saves the extracted features to a CSV file."""
    features.to_csv(file_path, index=False)

def main():
    """Main function to extract and save features."""
    # Load the preprocessed dataset
    data = pd.read_csv(PREPROCESSED_DATA_PATH)

    # Extract features
    features = extract_features(data)

    # Save extracted features
    save_features(features, FEATURES_PATH)

if __name__ == "__main__":
    main()
