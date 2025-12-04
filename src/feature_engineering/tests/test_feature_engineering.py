""" Tests for feature engineering functions. """

from unittest.mock import patch

import pytest
import polars as pl
import pandera as pa
from click.testing import CliRunner

from src.feature_engineering.feature_engineering import (
    compute_team_game_features,
    run,
)

@pytest.fixture
def pbp_df():
    """ Fixture for pbp data """
    return pl.DataFrame({
        "game_id": ["G1", "G1", "G1", "G1"],
        "posteam": ["BUF", "BUF", "KC", "KC"],

        "season_year": [2020, 2020, 2020, 2020],

        "epa": [1.0, -0.5, 0.3, 0.1],
        "success": [1, 0, 1, 0],
        "yards_gained": [20, 3, 17, -2],

        "drive_ended_with_score": [1, 0, 1, 0],
        "drive_yards_penalized": [5, 0, 10, 0],

        "interception": [0, 0, 1, 0],
        "fumble_lost": [0, 1, 0, 0],
        "penalty_yards": [10, 0, 15, 0],

        "third_down_converted": [1, 0, 0, 1],
        "third_down_failed": [0, 1, 1, 0],
        "fourth_down_converted": [0, 0, 0, 0],
        "fourth_down_failed": [0, 0, 0, 0],

        "pass_attempt": [1, 0, 1, 1],
        "rush_attempt": [0, 1, 0, 1],
        "yardline_100": [15, 30, 25, 18],  # red zone indicator

        "passing_yards": [20, None, 25, 5],
        "rushing_yards": [None, 7, None, -2],
        "complete_pass": [1, None, 1, 0],
        "touchdown": [1, 0, 0, 1],

        "posteam_type": ["home", "home", "away", "away"],
        "roof": ["indoors", "indoors", "outdoors", "outdoors"],
        "surface": ["turf", "turf", "grass", "grass"],
        "temp": [70, 70, 60, 60],
        "wind": [2, 2, 5, 5],

        "home_score": [14, 21, 10, 10],
        "away_score": [21, 21, 17, 24],
    })

def test_epa_and_success_logic(pbp_df):
    """ test epa and success rate calculations """
    result = compute_team_game_features(pbp_df)
    buf = result.filter(pl.col("posteam") == "BUF")

    # BUF EPA = [1.0, -0.5] → avg = 0.25, total = 0.5
    assert float(buf["avg_epa"][0]) == pytest.approx((1.0 + -0.5) / 2)
    assert float(buf["total_epa"][0]) == pytest.approx(0.5)

    # Success rate = [1, 0] → 0.5
    assert float(buf["success_rate"][0]) == pytest.approx(0.5)

def test_explosive_play_rate(pbp_df):
    """ test explosive play rate calculations """
    result = compute_team_game_features(pbp_df)

    # BUF: yards_gained = [20, 3] → 1 explosive out of 2 plays = 0.5
    buf = result.filter(pl.col("posteam") == "BUF")
    assert float(buf["pct_plays_over_15yds"][0]) == pytest.approx(0.5)

    # KC: yards_gained = [17, -2] → 1 explosive out of 2 plays = 0.5
    kc = result.filter(pl.col("posteam") == "KC")
    assert float(kc["pct_plays_over_15yds"][0]) == pytest.approx(0.5)

def test_turnover_count(pbp_df):
    """ test turnover count calculations """
    result = compute_team_game_features(pbp_df)

    # BUF: 0 INT + 1 fumble = 1
    buf = result.filter(pl.col("posteam") == "BUF")
    assert float(buf["turnover_count"][0]) == 1.0

    # KC: 1 INT + 0 fumble = 1
    kc = result.filter(pl.col("posteam") == "KC")
    assert float(kc["turnover_count"][0]) == 1.0

def test_down_success_rate(pbp_df):
    """ test down success rate calculations """
    result = compute_team_game_features(pbp_df)

    # For BUF:
    # Converted: 3rd_conv = 1, 4th_conv = 0 → converted = 1
    # Attempts = 3rd_conv + 3rd_fail + 4th_conv + 4th_fail = 1+1+0+0 = 2
    # Rate = 1/2 = 0.5
    buf = result.filter(pl.col("posteam") == "BUF")
    buf_rate = buf["third_down_success_rate"][0]
    if isinstance(buf_rate, pl.Series):
        buf_rate = buf_rate[0]
    assert float(buf_rate) == pytest.approx(0.5)

    # KC:
    # Converted: 1 (fourth_down_converted)
    # Attempts: 0+1+1+0 = 2
    # Rate = 1/2 = 0.5
    kc = result.filter(pl.col("posteam") == "KC")
    kc_rate = kc["third_down_success_rate"][0]
    if isinstance(kc_rate, pl.Series):
        kc_rate = kc_rate[0]
    assert float(kc_rate) == pytest.approx(0.5)

def test_passing_metrics(pbp_df):
    """ test passing metrics calculations """
    result = compute_team_game_features(pbp_df)
    buf = result.filter(pl.col("posteam") == "BUF")

    # BUF passing plays → pass_attempt = [1, 0]
    # EPA: [1.0] → mean = 1.0
    assert float(buf["pass_epa_mean"][0]) == pytest.approx(1.0)

    # Yards: [20] → avg = 20
    assert float(buf["pass_yards_avg"][0]) == pytest.approx(20.0)

    # Completion %: [1] → 1.0
    assert float(buf["comp_pct"][0]) == pytest.approx(1.0)

def test_rushing_metrics(pbp_df):
    """ test rushing metrics calculations """
    result = compute_team_game_features(pbp_df)
    buf = result.filter(pl.col("posteam") == "BUF")

    # BUF rushing plays → rush_attempt = [0,1]
    # EPA: [-0.5] → mean = -0.5
    assert float(buf["rush_epa_mean"][0]) == pytest.approx(-0.5)

    # Rushing yards: [7] → avg = 7
    assert float(buf["rush_yards_avg"][0]) == pytest.approx(7.0)

def test_redzone_td_rate(pbp_df):
    """ test red zone touchdown rate calculations """
    result = compute_team_game_features(pbp_df)
    buf = result.filter(pl.col("posteam") == "BUF")

    # BUF red zone plays → yardline_100 <= 20 → [15]
    # TD: [1] → mean = 1.0
    assert float(buf["td_rate_in_redzone"][0]) == pytest.approx(1.0)

def test_final_score_logic(pbp_df):
    """ test final score calculations """
    result = compute_team_game_features(pbp_df)
    buf = result.filter(pl.col("posteam") == "BUF")

    # BUF home_score = [14, 21] → max = 21
    assert buf["final_home_score"][0] == 21

    # BUF away_score = [21, 21] → max = 21
    assert buf["final_away_score"][0] == 21

def test_compute_team_game_features_basic(pbp_df):
    """ test basic functionality of compute_team_game_features """
    result = compute_team_game_features(pbp_df)

    assert result.shape[0] == 2
    assert set(result["posteam"]) == {"BUF", "KC"}

    expected_cols = {
        "avg_epa", "total_epa", "success_rate",
        "pct_plays_over_15yds", "pct_drives_scored",
        "avg_drive_yards_penalized", "turnover_count",
        "penalty_yards_total", "third_down_success_rate",
        "home_away_flag", "roof_type", "surface", "temp", "wind",
        "pass_epa_mean", "pass_yards_avg", "comp_pct",
        "rush_epa_mean", "rush_yards_avg",
        "td_rate_in_redzone",
        "final_home_score", "final_away_score",
    }

    assert expected_cols.issubset(result.columns)

    buf = result.filter(pl.col("posteam") == "BUF")
    assert float(buf["total_epa"][0]) == pytest.approx(1.0 + -0.5)

@patch("src.feature_engineering.feature_engineering.TeamGameFeaturesSchema.validate") # pylint: disable=W0613
@patch("src.feature_engineering.feature_engineering.RawPlayByPlaySchema.validate") # pylint: disable=W0613
@patch("src.feature_engineering.feature_engineering.load_data")
@patch("src.feature_engineering.feature_engineering.save_data")
@patch("src.feature_engineering.feature_engineering.compute_team_game_features")
def test_run_cli_success(mock_compute, mock_save, mock_load, mock_raw_schema, mock_team_schema): # pylint: disable=W0613
    """Test the Click CLI (run) success path."""

    mock_load.return_value = pl.DataFrame({"game_id": ["2020_1"], "posteam": ["BUF"], "season_year": [2020]})
    mock_compute.return_value = pl.DataFrame({"game_id": ["2020_1"], "posteam": ["BUF"], "season_year": [2020]})

    runner = CliRunner()
    result = runner.invoke(
        run,
        [
            "--local",
            "--input-path", ".",
            "--output-path", ".",
            "--filename", "test.parquet",
            "--year", "2020",
        ],
    )

    assert result.exit_code == 0
    mock_load.assert_called()
    mock_compute.assert_called_once()
    mock_save.assert_called_once()

@patch("src.feature_engineering.feature_engineering.TeamGameFeaturesSchema.validate") # pylint: disable=W0613
@patch("src.feature_engineering.feature_engineering.RawPlayByPlaySchema.validate") # pylint: disable=W0613
@patch("src.feature_engineering.feature_engineering.load_data")
@patch("src.feature_engineering.feature_engineering.compute_team_game_features")
def test_run_cli_schema_error(mock_compute, mock_load, mock_raw_schema, mock_team_schema):  # pylint: disable=W0613
    """Test the CLI handles Pandera schema errors without crashing."""

    mock_load.return_value = pl.DataFrame({"game_id": [], "posteam": []})

    mock_compute.side_effect = pa.errors.SchemaError(
        schema=None,
        data=pl.DataFrame({"game_id": [], "posteam": []}),
        message="Schema error!"
    )

    runner = CliRunner()
    result = runner.invoke(
        run,
        [
            "--local",
            "--input-path", ".",
            "--output-path", ".",
            "--filename", "test.parquet",
        ],
    )

    assert result.exit_code == 1
