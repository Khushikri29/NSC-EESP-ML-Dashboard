import joblib

# Load only existing model
load_model = joblib.load("models/load_model.pkl")

def predict_all(app_phase):
    prediction = load_model.predict([[app_phase]])
    return {
        "predicted_load": float(prediction[0])
    }
