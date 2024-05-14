import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from src.utils import load_object

# Define the request body structure
class PredictionRequest(BaseModel):
    Buying_Price: str
    Maintenance_Price: str
    No_of_Doors: str
    Person_Capacity: str
    Size_of_Luggage : str 
    Safety  : str


# Initialize FastAPI app
app = FastAPI()

# ids to labels
idx_to_labels={ 0:'unacc',
                1:'acc'  ,
                2:'good' ,
                3:'vgood'}

@app.get("/")
def read_root():
    return {"message": "Welcome to the prediction API"}

@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        data_dict=request.dict()
        data_df=pd.DataFrame([data_dict])
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        model = joblib.load(model_path)
        # Convert request data to the appropriate format for prediction
        preprocessor=load_object(file_path=preprocessor_path)
        data_scaled=preprocessor.transform(data_df)
        # Make prediction
        prediction = model.predict(data_scaled)

        return f"The Prediction is :{idx_to_labels[prediction[0]]}"
        # # Return prediction result
        # return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app, use the following command in the terminal:
# uvicorn main:app --reload
