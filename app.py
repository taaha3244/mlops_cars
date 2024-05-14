import os
from src.exception import CustomException
from src.logger import logging
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import joblib
import pandas as pd
import logging



# Define the request body structure using Enums for input validation
class Price(str, Enum):
    low = "low"
    med = "med"
    high = "high"
    vhigh = "vhigh"

class Doors(str, Enum):
    two = "2"
    three = "3"
    four = "4"
    five_more = "5more"

class Capacity(str, Enum):
    two = "2"
    four = "4"
    more = "more"

class Luggage(str, Enum):
    small = "small"
    med = "med"
    big = "big"

class Safety(str, Enum):
    low = "low"
    med = "med"
    high = "high"

class PredictionRequest(BaseModel):
    Buying_Price: Price
    Maintenance_Price: Price
    No_of_Doors: Doors
    Person_Capacity: Capacity
    Size_of_Luggage: Luggage
    Safety: Safety

# Initialize FastAPI app
app = FastAPI()

# Load model and preprocessor
model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# Index to labels mapping
idx_to_labels = {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}

@app.get("/", response_model=dict)
def read_root():
    return {"message": "Welcome to the prediction API"}

@app.post("/predict", response_model=dict)
def predict(request: PredictionRequest):
    try:
        data_df = pd.DataFrame([request.dict()])
        data_scaled = preprocessor.transform(data_df)
        prediction = model.predict(data_scaled)
        return {"prediction": idx_to_labels[int(prediction[0])]}
    except Exception as e:
        raise CustomException(e,sys)
            
