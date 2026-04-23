from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Request schema
class AnimalData(BaseModel):
    animal_type: str
    age: float
    weight: float
    milk: float
    location: str

# ✅ Encode input (simple example)
def preprocess(data):
    animal_map = {"buffalo": 0, "cow": 1, "goat": 2, "sheep": 3, "poultry": 4}
    location_map = {
        "hyderabad": 0,
        "mumbai": 1,
        "bangalore": 2,
        "delhi": 3,
        "chennai": 4,
        "pune": 5,
        "kolkata": 6,
    }

    animal = animal_map.get(data.animal_type.lower(), 0)
    location = location_map.get(data.location.lower(), 0)

    return np.array([[animal, data.age, data.weight, data.milk, location]])

# ✅ Prediction API
@app.post("/predict")
def predict(data: AnimalData):
    features = preprocess(data)
    prediction = model.predict(features)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }

# ✅ Health check
@app.get("/")
def root():
    return {"message": "ML API is running 🚀"}
