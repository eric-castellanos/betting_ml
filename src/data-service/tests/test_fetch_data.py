import pytest
from unittest.mock import patch
import polars as pl

from ..fetch_data import fetch_play_by_play_data

@pytest.mark.parametrize(
    "years, urls, expected_keys",
    [
        (
            ["2020"],
            ["https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2020.parquet"],
            ["play_by_play_2020"]
        ),
        (
            ["2021", "2022"],
            [
                "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.parquet",
                "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2022.parquet"
            ],
            ["play_by_play_2021", "play_by_play_2022"]
        ),
    ]
)
def test_fetch_play_by_play_data(years, urls, expected_keys):
    dummy_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with patch("polars.read_parquet", return_value=dummy_df):
        result = fetch_play_by_play_data(years, urls)
        assert isinstance(result, dict)
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pl.DataFrame)
            assert result[key].shape == dummy_df.shape