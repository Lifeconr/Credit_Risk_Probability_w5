from pydantic import BaseModel

class PredictionRequest(BaseModel):
    """
    Pydantic model for the input data for a single prediction.
    The features must match the columns of X_processed.
    """
    Recency: float
    AvgTransactionAmount: float
    StdTransactionAmount: float
    AvgTransactionHour: float
    Frequency_Bin_woe: float
    Monetary_Bin_woe: float

class PredictionResponse(BaseModel):
    """Pydantic model for the prediction response."""
    risk_probability: float