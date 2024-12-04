from flask import Flask, render_template, request
import joblib  # For loading the pretrained model
import pandas as pd  # For handling input data

app = Flask(__name__)

# Load the pretrained model
model = joblib.load("model.pkl")  # Replace with the actual path to your model file

@app.route("/")
def index():
    """
    Serve the homepage with the input form.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form submission, process input data, and return prediction results.
    """
    try:
        # Get user input from the form
        input_data = request.form.get("input")
        
        # Convert input data into a DataFrame for compatibility with the model
        # Assuming the input is a single feature; adjust columns as needed
        input_df = pd.DataFrame([[input_data]], columns=["feature_name"])  # Replace "feature_name" with the actual column name

        # Preprocess input data if needed
        # For example, scaling or encoding can be done here
        # preprocessed_data = preprocess(input_df)  # Uncomment and define preprocess if required
        
        # Make prediction using the loaded model
        prediction = model.predict(input_df)  # Ensure input_df matches the model's expected format

        # Format the prediction result for display
        predicted_value = prediction[0]  # Assuming a single output value
        
        # Render the result in the result.html template
        return render_template("result.html", prediction=predicted_value)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)