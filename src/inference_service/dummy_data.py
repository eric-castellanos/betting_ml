from src.inference_service.schemas import PredictionRequest


def build_dummy_features(request: PredictionRequest) -> dict:
    """
    Build placeholder feature values for inference.
    Replace with real feature engineering (recent games, injuries, etc.).
    """
    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "game_id": request.game_id or "unknown_game",
        "location": request.location or "unknown",
        "vegas_spread": request.spread if request.spread is not None else 0.0,
        # Dummy numeric features to satisfy model input shape
        "home_offense_rating": 0.5,
        "away_offense_rating": 0.5,
        "home_defense_rating": 0.5,
        "away_defense_rating": 0.5,
        "home_qb_health": 1.0,
        "away_qb_health": 1.0,
    }
