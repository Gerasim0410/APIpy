from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
model = joblib.load("age_model_svr.pkl")

class FeatureSet(BaseModel):
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
async def predict(features: FeatureSet):
    order = [
        'cam_aver_time', 'cam_aver_abs_delta',
        'math_aver_time', 'math_part_correct',
        'mem_aver_time', 'mem_part_correct',
        'mun_aver_time',
        'str_aver_time', 'str_part_correct',
        'swa_aver_time', 'swa_part_correct',
        'sex'
    ]
    x = np.array([[getattr(features, key) for key in order]])
    age = float(model.predict(x)[0])
    return { "age": age }
