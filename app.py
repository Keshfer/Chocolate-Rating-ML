from flask import Flask, render_template, request
import joblib  # For loading the pretrained model
import pandas as pd  # For handling input data

app = Flask(__name__)

# Load the pretrained model
model_lightgbm = joblib.load("lightgbm_model.pkl")  # Replace with the actual path to your model file
model_svr = joblib.load("SVR_model.pkl")
#grid2 only uses REf and Cocoa Percent grid1 uses all the features
model_rf = joblib.load("rfmodel_grid.joblib")
model_rf2 = joblib.load("rfmodel_grid2.joblib") 
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
        #print("form submission:", input_data)
        
        # Convert input data into a DataFrame for compatibility with the model
        formatted_submission = []
        #print(f"formatted list {formatted_submission}")
        #print(f"dataframe {input_df}")
        if input_data.get("model_type") == 'LightGBM':
            curr_model = model_lightgbm
        elif input_data.get("model_type") == "SVR":
            curr_model = model_svr
        elif input_data.get("model_type") == 'Random Forest':
            curr_model = model_rf
        elif input_data.get("model_type") == 'Random Forest 2':
            curr_model = model_rf2
        if input_data.get('model_type') != 'Random Forest 2':
            formatted_submission.append(int(input_data.get("REF"))) 
            formatted_submission.append(int(input_data.get("date"))) 
            formatted_submission.append(float(input_data.get("cocoa_percent"))) 
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
            formatted_submission = [formatted_submission]
            input_df = pd.DataFrame(formatted_submission, columns=["REF","Review_Date","Cocoa_Percent","Company_Location_AF","Company_Location_AS","Company_Location_CA","Company_Location_CEU","Company_Location_CR","Company_Location_EEU","Company_Location_NA","Company_Location_OC","Company_Location_SA","Company_Location_WEU","Broad_Bean_Origin_AF","Broad_Bean_Origin_AS","Broad_Bean_Origin_CA","Broad_Bean_Origin_CR","Broad_Bean_Origin_NA","Broad_Bean_Origin_OC","Broad_Bean_Origin_SA","Broad_Bean_Origin_Unknown"])  # Replace "feature_name" with the actual column name
        else:
            formatted_submission.append(input_data.get("REF"))
            formatted_submission.append(input_data.get("cocoa_percent")) 
            formatted_submission = [formatted_submission]
            input_df = pd.DataFrame(formatted_submission, columns=["REF", "Cocoa Percent"])
        # Preprocess input data if needed
        # For example, scaling or encoding can be done here
        # preprocessed_data = preprocess(input_df)  # Uncomment and define preprocess if required
        
        # Make prediction using the loaded model
        prediction = curr_model.predict(input_df)  # Ensure input_df matches the model's expected format

        # Format the prediction result for display
        predicted_value = prediction[0]  # Assuming a single output value
        #print(predicted_value)
        
        # Render the result in the result.html template
        return render_template("results.html", prediction=predicted_value)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)