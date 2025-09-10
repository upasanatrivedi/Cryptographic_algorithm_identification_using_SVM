from flask import Flask, render_template, request, jsonify
import sys
import os

# Add the svm_classifiers directory to the path for importing central_analysis.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function from central_analysis.py
from central_analysis import classify_ciphertext

app = Flask(__name__)

# Route to render the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle prediction requests
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the raw ciphertext from the user input
        data = request.get_json()
        ciphertext = data.get("ciphertext", "")

        # Get the prediction from the central_analysis function
        final_prediction = classify_ciphertext(ciphertext)

        # Return the prediction in JSON format
        return jsonify({"predicted_algorithm": final_prediction})
    
    except Exception as e:
        print("Error in prediction:", e)  # Log the error to console for debugging
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
