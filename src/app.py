from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# تحميل الموديل
model = pickle.load(open("artifacts/model.pkl", "rb"))

# تعريف شكل البيانات المتوقعة
class Features(BaseModel):
    values: list

@app.post("/predict")
def predict(data: Features):
    prediction = model.predict([data.values])
    return {"prediction": int(prediction[0])}
