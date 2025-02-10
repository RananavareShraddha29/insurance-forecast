from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from prophet import Prophet

# Load the saved model
with open("prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Define available states
AVAILABLE_STATES = ['TO', 'SP', 'RJ', 'MG', 'BA']

@app.route('/')
def home():
    # Render the HTML page for user input
    return render_template("index.html", states=AVAILABLE_STATES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the form
        state = request.form.get("state")
        date_input = request.form.get("date")  # Expected format: "YYYY-MM-DD"

        # Ensure input is a valid date
        date_df = pd.DataFrame({"ds": [pd.to_datetime(date_input)]})

        # Placeholder for state-based processing (if needed)
        # Example: Adjust predictions based on the state
        # You can use additional logic here if premiums vary by state

        # Generate prediction
        forecast = model.predict(date_df)
        predicted_value = forecast["yhat"].values[0]

        # Return the prediction
        return render_template(
            "result.html",
            state=state,
            date=date_input,
            predicted_premium=round(predicted_value, 2)
        )

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
