from fastapi import FastAPI
import pandas as pd
import mlflow
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API to predict credit risk probability for a customer.",
    version="1.0.0"
)

# Load the registered model from MLflow
MODEL_URI = "models:/CreditRiskClassifier/latest"
model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Credit Risk API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts customer data and returns the risk probability.
    """
    # Convert the request data into a pandas DataFrame
    data = pd.DataFrame([request.dict()])
    
    # Predict the probability of being high-risk (class 1)
    probability = model.predict_proba(data)[:, 1][0]
    
    return PredictionResponse(risk_probability=probability)