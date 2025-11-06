from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(root_path="/APIpy")

model = joblib.load("Bagging_age_predictor.joblib")

class Features(BaseModel):
    swallow_time_red: float
    munster_mean_attempt_time: float
    stroop_var_attempt_time: float

@app.post("/predict")
async def predict(features: Features):
    data = pd.DataFrame([features.dict()])
    pred = float(model.predict(data)[0])
    return {"age": pred}
