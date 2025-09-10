# cryptographic-algorithm-identifier-using-SVM
An SVM-based classifier served via Flask/FastAPI to automatically identify crypto algorithms from raw ciphertext.

# Cryptographic Algorithm Identification

This project is a *machine learning–based system* that automates the identification of cryptographic algorithms from ciphertext.
It was developed as part of the *Software Engineering Lab project* in the *Department of Computer Science, Indraprastha College for Women (University of Delhi).*


## Overview

The system processes ciphertext datasets and classifies the underlying cryptographic algorithm using *Support Vector Machines (SVM)* with a *One-vs-One classifier voting mechanism*.
It is designed for applications in *cybersecurity, blockchain, and cryptanalysis*, reducing manual effort in algorithm recognition.

## Features

* Identification of cryptographic algorithms from ciphertext
* Machine learning pipeline with preprocessing, feature extraction, and classification
* Web interface for dataset upload and prediction results
* Backend built with Flask/FastAPI
* Visualization of results using Matplotlib/Plotly
* Scalable and modular design for adding new algorithms

## Tech Stack

* *Programming Language*: Python
* *Libraries*: NumPy, Pandas, Scikit-learn, Joblib, Flask/FastAPI
* *Frontend*: HTML, CSS, JavaScript
* *Visualization*: Matplotlib, Plotly

## How It Works

1. *Data Preprocessing* – Clean and normalize input ciphertext.
2. *Feature Extraction* – Extract length, entropy, and frequency distribution.
3. *Model Training* – Train binary SVM classifiers for algorithm pairs.
4. *Voting Mechanism* – Combine predictions with One-vs-One classification.
5. *Deployment* – Provide predictions through an API and simple web interface.


## Installation & Usage

1. Clone the repository:

   bash
   git clone https://github.com/your-username/cryptographic-algorithm-identification.git
   cd cryptographic-algorithm-identification
   
2. Install dependencies:

   bash
   pip install -r requirements.txt
   
3. Run the backend:

   bash
   python backend/app.py
   
4. Open the web interface in your browser at:

   
   http://127.0.0.1:5000/
   


## Testing

* *Unit tests* for preprocessing, feature extraction, and classifiers
* *Integration tests* for full pipeline validation
* *Performance tests* to measure accuracy and scalability


## Future Scope

* Support for more cryptographic algorithms (ChaCha20, ECC, hashing methods)
* Integration of deep learning models for higher accuracy
* Real-time prediction capabilities
* Cross-platform deployment (desktop and mobile)
* Enhanced visual analytics and secure authentication

## Authors

* *Upasana Trivedi* (22/CS/74)
* *Prakriti Shree* (22/CS/49)

*Supervisor:* Mrs. Vimala Parihar
Department of Computer Science
Indraprastha College for Women, University of Delhi
