
import pytest
import polars as pl
from unittest.mock import patch

from src.data_service.fetch_data import fetch_play_by_play_data

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
def test_fetch_play_by_play_data_valid(years, urls, expected_keys):
    dummy_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with patch("polars.read_parquet", return_value=dummy_df):
        result = fetch_play_by_play_data(years, urls)
        assert isinstance(result, dict)
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pl.DataFrame)
            assert result[key].shape == dummy_df.shape


@pytest.mark.parametrize(
    "years, urls, exception, message",
    [
        (["2020"], [], ValueError, "Number of years must match number of URLs."),
        ([2020], ["url"], ValueError, "Year must be passed a string."),
        (["2020"], ["not_a_url"], ValueError, "Invalid URL: URL must start with either http:// or https://.")
    ]
)
def test_fetch_play_by_play_data_invalid(years, urls, exception, message):
    with pytest.raises(exception) as excinfo:
        fetch_play_by_play_data(years, urls)
    assert message in str(excinfo.value)

def test_fetch_play_by_play_data_oserror():
    with patch("src.data_service.fetch_data.pl.read_parquet", side_effect=OSError("mocked failure")):
        years = ["2020"]
        urls = ["http://valid-url.parquet"]

        with pytest.raises(OSError) as excinfo:
            fetch_play_by_play_data(years, urls)

        assert "Network or file error for http://valid-url.parquet" in str(excinfo.value)