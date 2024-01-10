from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the pre-trained Random Forest model and StandardScaler
model = load("random_forest_model.joblib")
scaler = load("scaler.joblib")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input parameters from the form
        input_data = {
            "Pregnancies": int(request.form["pregnancies"]),
            "Glucose": int(request.form["glucose"]),
            "BloodPressure": int(request.form["blood_pressure"]),
            "SkinThickness": int(request.form["skin_thickness"]),
            "Insulin": int(request.form["insulin"]),
            "BMI": float(request.form["bmi"]),
            "DiabetesPedigreeFunction": float(
                request.form["diabetes_pedigree_function"]
            ),
            "Age": int(request.form["age"]),
        }

        # Preprocess the input data
        input_df = pd.DataFrame([input_data])
        input_df_scaled = scaler.transform(
            input_df
        )  # Use the same scaler used during training

        # Make predictions using the pre-trained model
        prediction = model.predict(input_df_scaled)

        # Display the prediction result
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
