# Titanic Survival Predictor 🚢

ML model that predicts Titanic passenger survival with **83% accuracy**.

## Tech Stack
- Python, Pandas, Scikit-Learn
- Random Forest Classifier
- FastAPI REST API

## How it works
Send a POST request to `/predict` with passenger details
and get survival prediction with probability.

## Run Locally
pip install -r requirements.txt
uvicorn app:app --reload
```
