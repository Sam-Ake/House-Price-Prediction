from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# Load model pipeline
try:
    model = joblib.load("house_price_model.pkl")
except Exception as e:
    model = None

@app.get("/")
def read_root():
    return {"status": "API is online"}

@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Model pickle file not loaded.")
    
    try:
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])
        
        # Calculate engineered feature expected by the model
        if "house_age" not in input_df.columns:
            input_df["house_age"] = input_df["sale_year"] - input_df["yr_built"]
            
        # Predict price
        prediction = model.predict(input_df)[0]
        
        return {
            "status": "success",
            "predicted_price": round(float(prediction), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))