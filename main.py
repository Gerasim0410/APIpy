from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
model = joblib.load("age_model_svr.pkl")

class Features(BaseModel):
    cam_aver_time: float
    cam_aver_abs_delta: float
    math_aver_time: float
    math_part_correct: float
    mem_aver_time: float
    mem_part_correct: float
    mun_aver_time: float
    str_aver_time: float
    str_part_correct: float
    swa_aver_time: float
    swa_part_correct: float
    sex: int

@app.post("/predict")
def predict_age(features: Features):
    try:
        x = np.array([[getattr(features, k) for k in features.__fields__]])
        age = model.predict(x)[0]
        return {"age": float(age)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
