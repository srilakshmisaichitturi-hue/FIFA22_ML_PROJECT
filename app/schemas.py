from pydantic import BaseModel, Field

class PlayerStats(BaseModel):
    age: int = Field(default=25, ge=15, le=55)
    height_cm: float = Field(default=182.0, ge=140.0, le=220.0)
    weight_kg: float = Field(default=76.0, ge=45.0, le=130.0)
    pace: float = Field(default=78.0, ge=0.0, le=100.0)
    shooting: float = Field(default=72.0, ge=0.0, le=100.0)
    passing: float = Field(default=75.0, ge=0.0, le=100.0)
    dribbling: float = Field(default=77.0, ge=0.0, le=100.0)
    defending: float = Field(default=60.0, ge=0.0, le=100.0)
    physic: float = Field(default=72.0, ge=0.0, le=100.0)
    attacking_crossing: float = Field(default=70.0, ge=0.0, le=100.0)
    attacking_finishing: float = Field(default=68.0, ge=0.0, le=100.0)

class PredictionResponse(BaseModel):
    predicted_overall: float
    status: str