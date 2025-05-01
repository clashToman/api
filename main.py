from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("classifier.pkl")

# Encodings (must match training)
soil_type_mapping = {
    "black": 0,
    "clayey": 1,
    "loamy": 2,
    "red": 3,
    "sandy": 4
}

crop_type_mapping = {
    "barley": 0,
    "cotton": 1,
    "ground nuts": 2,
    "maize": 3,
    "millets": 4,
    "oil seeds": 5,
    "paddy": 6,
    "pulses": 7,
    "sugarcane": 8,
    "tobacco": 9,
    "wheat": 10
}

fertilizer_mapping = {
    0: "10-26-26",
    1: "14-35-14",
    2: "17-17-17",
    3: "20-20",
    4: "28-28",
    5: "DAP",
    6: "Urea"
}

# Request schema
class FertilizerRequest(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    nitrogen: float
    phosphorous: float
    potassium: float

# Prediction endpoint
@app.post("/predict")
def predict(data: FertilizerRequest):
    try:
        # Encode categorical inputs
        soil_encoded = soil_type_mapping.get(data.soil_type.lower())
        crop_encoded = crop_type_mapping.get(data.crop_type.lower())

        if soil_encoded is None or crop_encoded is None:
            raise ValueError("Invalid soil or crop type")

        features = [[
            data.temperature,
            data.humidity,
            data.moisture,
            soil_encoded,
            crop_encoded,
            data.nitrogen,
            data.potassium,
            data.phosphorous
        ]]

        # Get prediction and map to label
        prediction = model.predict(features)[0]
        fertilizer_name = fertilizer_mapping.get(int(prediction), "Unknown")

        # Prepare and save log
        recommendation = {
            "recommended_fertilizer": fertilizer_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open("recommendations_log.json", "a") as log_file:
            json.dump(recommendation, log_file)
            log_file.write("\n")

        return recommendation

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
