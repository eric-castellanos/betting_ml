from datetime import datetime

from src.inference_service.schemas import PredictionRequest


def build_dummy_features(request: PredictionRequest) -> dict:
    """
    Build placeholder numeric features that match the XGBoost model inputs.
    The request's team/location fields are accepted for future feature engineering
    but are not used by the current dummy feature set.
    """
    current_year = datetime.utcnow().year
    roof_outdoors = 1.0  # assume outdoor stadium if unknown

    return {
        "season_year": float(current_year),
        "avg_epa": 0.0,
        "success_rate": 0.0,
        "pct_drives_scored": 0.0,
        "avg_drive_yards_penalized": 0.0,
        "total_epa": 0.0,
        "penalty_yards_total": 0.0,
        "temp": 70.0,
        "wind": 5.0,
        "pass_epa_mean": 0.0,
        "pass_yards_avg": 0.0,
        "comp_pct": 0.5,
        "rush_epa_mean": 0.0,
        "rush_yards_avg": 0.0,
        "td_rate_in_redzone": 0.0,
        "pct_plays_over_15yds": 0.0,
        "turnover_count": 0.0,
        "third_down_success_rate": 0.0,
        "is_home": 1.0,
        "is_turf": 0.0,
        "posteam_id": 0.0,
        "roof_type_open": 0.0,
        "roof_type_closed": 0.0,
        "roof_type_dome": 0.0,
        "roof_type_outdoors": roof_outdoors,
    }
