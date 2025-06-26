from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("obesity_model.pkl")
le = joblib.load("label_encoder.pkl") 

# Input schema
class InputData(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Obesity Prediction API"}

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])

    prediction_encoded = model.predict(input_df)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]

    return {"prediction": prediction_label}
