from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# App aur model load karo
app = FastAPI()
model = joblib.load("titanic_model.pkl")

# Input format define karo
class Passenger(BaseModel):
    Pclass: int
    Sex: int        # 0 = male, 1 = female
    Age: float
    Fare: float
    Embarked: int   # 0=S, 1=C, 2=Q
    FamilySize: int
    IsAlone: int

# Home route
@app.get("/")
def home():
    return {"message": "Titanic Survival Predictor API is running!"}

# Prediction route
@app.post("/predict")
def predict(passenger: Passenger):
    data = [[
        passenger.Pclass,
        passenger.Sex,
        passenger.Age,
        passenger.Fare,
        passenger.Embarked,
        passenger.FamilySize,
        passenger.IsAlone
    ]]
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    return {
        "survived": int(prediction),
        "result": "Survived ✅" if prediction == 1 else "Did not survive ❌",
        "probability": round(float(probability), 2)
    }