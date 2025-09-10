import pandas as pd
import os
from config import RAW_DATA_PATH, DATA_DIR

def load_dataset(file_path):
    return pd.read_csv(file_path)

def drop_unnecessary_columns(data):
    """Removes columns that are not useful for classification."""
    columns_to_drop = ['S.No.', 'Plaintext', 'AES_Key', 'DES_Key', 'Blowfish_Key', 'RSA_public_key', 'RSA_private_key']
    return data.drop(columns=columns_to_drop, errors="ignore")

def drop_missing_values(data):
    """Drops rows with missing values."""
    return data.dropna()

def save_preprocessed_data(data, output_file):
    """Saves the preprocessed dataset to the specified output file."""
    output_path = os.path.join(DATA_DIR, output_file)
    data.to_csv(output_path, index=False)

def preprocess_dataset(data: pd.DataFrame):  
    # Preprocess the dataset
    data = drop_unnecessary_columns(data)
    data = drop_missing_values(data)
    return data

if __name__ == "__main__":
    data=load_dataset(RAW_DATA_PATH)
    pre_data=preprocess_dataset(data)
    save_preprocessed_data(pre_data,"preprocessed_ciphers.csv")
