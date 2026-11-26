"""
app/schemas.py

Pydantic schemas for the MP076 ATM Cash Demand prediction service.
Enforces client validation rules:
1. other_atms_1km above 25 is rejected (Field(..., le=25))
2. distance_to_market_km below 0 is rejected (Field(..., ge=0))
"""

from typing import Literal, Union
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request schema for POST /predict.
    Must contain exactly the 8 required features.
    """
    nearby_shops: int = Field(..., description="Number of shops within 500m")
    days_to_salary: int = Field(..., description="Days to the 1st of the month")
    footfall_index: int = Field(..., description="Street footfall index")
    distance_to_market_km: float = Field(
        ...,
        ge=0,
        description="Distance to main market in km (must be >= 0)",
    )
    avg_ticket_rs: Union[int, float] = Field(..., description="Average withdrawal last week in rupees")
    other_atms_1km: int = Field(
        ...,
        le=25,
        description="Number of other ATMs within 1km (must be <= 25)",
    )
    atm_type: str = Field(..., description="ATM placement: standalone, bank_branch, mall")
    area: str = Field(..., description="Area type: Market, Residential, Highway")

    model_config = {
        "json_schema_extra": {
            "example": {
                "nearby_shops": 55,
                "days_to_salary": 14,
                "footfall_index": 62,
                "distance_to_market_km": 2.8,
                "avg_ticket_rs": 2400,
                "other_atms_1km": 8,
                "atm_type": "standalone",
                "area": "Residential",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""
    prediction: float = Field(..., description="Predicted withdrawal in thousand rupees")


class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str = Field(default="ok", description="Service health status")
