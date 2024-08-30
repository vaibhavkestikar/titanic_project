# main.py
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the trained model
model = joblib.load("titanic_model.pkl")

@app.post("/predict")
def predict(Pclass: int, Sex: int, Age: float, SibSp: int, Parch: int, Fare: float, Embarked: int):
    # Create a DataFrame for the input features
    data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    })
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Return the result
    return {"Survived": int(prediction[0])}
