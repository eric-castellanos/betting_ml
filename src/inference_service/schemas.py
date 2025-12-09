from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team abbreviation")
    away_team: str = Field(..., description="Away team abbreviation")
    game_id: str | None = Field(None, description="Unique game identifier")
    location: str | None = Field(None, description="Location or stadium")
    spread: float | None = Field(None, description="Market spread if available")

    @field_validator("spread")
    @classmethod
    def validate_spread(cls, v: float | None) -> float | None:
        if v is not None and (v < -50 or v > 50):
            raise ValueError("spread must be between -50 and 50")
        return v


class PredictionResponse(BaseModel):
    predicted_spread: float = Field(..., description="Model-predicted home minus away spread")
    model_version: str | None = Field(None, description="Model run_id or version identifier")
    input: PredictionRequest = Field(..., description="Echo of the input payload")
