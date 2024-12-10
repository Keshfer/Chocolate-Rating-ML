from flask import Flask, render_template, request
import joblib  # For loading the pretrained model
import pandas as pd  # For handling input data

app = Flask(__name__)

# Load the pretrained model
model = joblib.load("lightgbm_model.pkl")  # Replace with the actual path to your model file

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
        input_data = request.form
        print("form submission:", input_data)
        
        # Convert input data into a DataFrame for compatibility with the model
        formatted_submission = []
        formatted_submission.append(input_data.get("REF")) 
        formatted_submission.append(input_data.get("date")) 
        formatted_submission.append(input_data.get("cocoa_percent")) 
        #company location
        formatted_submission.append(int(input_data.get("company_location") == 'Africa'))
        formatted_submission.append(int(input_data.get("company_location") == 'Asia'))
        formatted_submission.append(int(input_data.get("company_location") == 'Central America'))
        formatted_submission.append(int(input_data.get("company_location") == 'Central Europe'))
        formatted_submission.append(int(input_data.get("company_location") == 'Carribean'))
        formatted_submission.append(int(input_data.get("company_location") == 'Eastern Europe'))
        formatted_submission.append(int(input_data.get("company_location") == 'North America'))
        formatted_submission.append(int(input_data.get("company_location") == 'Oceania'))
        formatted_submission.append(int(input_data.get("company_location") == 'South America'))
        formatted_submission.append(int(input_data.get("company_location") == 'Western Europe'))
        #Bean origin
        formatted_submission.append(int(input_data.get("bean_origin") == 'Africa'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'Asia'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'Central America'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'Carribean'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'North America'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'Oceania'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'South America'))
        formatted_submission.append(int(input_data.get("bean_origin") == 'Unknown'))
        input_df = pd.DataFrame(formatted_submission, columns=["REF","Review Date","Cocoa Percent","Rating","Company Location_AF","Company Location_AS","Company Location_CA","Company Location_CEU","Company Location_CR","Company Location_EEU","Company Location_NA","Company Location_OC","Company Location_SA","Company Location_WEU","Broad Bean Origin_AF","Broad Bean Origin_AS","Broad Bean Origin_CA","Broad Bean Origin_CR","Broad Bean Origin_NA","Broad Bean Origin_OC","Broad Bean Origin_SA","Broad Bean Origin_Unknown"])  # Replace "feature_name" with the actual column name
        #print(formatted_submission)
        

        # Preprocess input data if needed
        # For example, scaling or encoding can be done here
        # preprocessed_data = preprocess(input_df)  # Uncomment and define preprocess if required
        
        # Make prediction using the loaded model
        prediction = model.predict(input_df)  # Ensure input_df matches the model's expected format

        # Format the prediction result for display
        # predicted_value = prediction[0]  # Assuming a single output value
        
        # Render the result in the result.html template
        return render_template("result.html", prediction=3)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)