from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "NeuroPulse FastAPI Backend"


class PredictResponse(BaseModel):
    model_name: str
    prediction: int = Field(..., description="0=Real, 1=Deepfake")
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=100.0)
    details: dict = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    available_models: list[str]
