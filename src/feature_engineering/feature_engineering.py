import polars as pl

import logging

from src.utils.utils import save_data_s3, load_data_s3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(level)s - %(message)s - %(lineno)d")

logger = logging.getLogger(__name__)

import polars as pl

def compute_team_game_features(df: pl.DataFrame) -> pl.DataFrame:
    # Pre-filter for subsets you need just once
    pass_df = df.filter(pl.col("pass_attempt") == 1)
    rush_df = df.filter(pl.col("rush_attempt") == 1)
    redzone_df = df.filter(pl.col("yardline_100") <= 20)

    # Base groupby
    grouped = df.group_by(["game_id", "posteam"]).agg([
        # --- Scoring & Efficiency ---
        pl.col("epa").mean().alias("avg_epa"),
        pl.col("epa").sum().alias("total_epa"),
        pl.col("success").mean().alias("success_rate"),

        # --- Explosiveness ---
        (pl.col("yards_gained") > 15).mean().alias("pct_plays_over_15yds"),

        # --- Drive outcomes ---
        pl.col("drive_ended_with_score").mean().alias("pct_drives_scored"),
        pl.col("drive_yards_penalized").mean().alias("avg_drive_yards_penalized"),
        pl.col("drive_time_of_possession").mean().alias("drive_time_mean"),

        # --- Turnovers & Penalties ---
        (pl.col("interception") + pl.col("fumble_lost")).sum().alias("turnover_count"),
        pl.col("penalty_yards").sum().alias("penalty_yards_total"),

        # --- Down success ---
        (
            (
                pl.col("third_down_converted") + pl.col("fourth_down_converted")
            ).sum() /
            (
                pl.col("third_down_converted") + pl.col("third_down_failed") +
                pl.col("fourth_down_converted") + pl.col("fourth_down_failed")
            ).sum()
        ).alias("third_down_success_rate"),

        # --- Situational ---
        pl.col("posteam_type").first().alias("home_away_flag"),
        pl.col("roof").first().alias("roof_type"),
        pl.col("surface").first().alias("surface"),
        pl.col("temp").first().alias("temp"),
        pl.col("wind").first().alias("wind"),
    ])

    # --- Merge in specialized subsets (passing, rushing, redzone) ---
    pass_stats = pass_df.group_by(["game_id", "posteam"]).agg([
        pl.col("epa").mean().alias("pass_epa_mean"),
        pl.col("passing_yards").mean().alias("pass_yards_avg"),
        pl.col("complete_pass").mean().alias("comp_pct"),
    ])

    rush_stats = rush_df.group_by(["game_id", "posteam"]).agg([
        pl.col("epa").mean().alias("rush_epa_mean"),
        pl.col("rushing_yards").mean().alias("rush_yards_avg"),
    ])

    redzone_stats = redzone_df.group_by(["game_id", "posteam"]).agg([
        pl.col("touchdown").mean().alias("td_rate_in_redzone"),
    ])

    # --- Merge them all together ---
    result = (
        grouped
        .join(pass_stats, on=["game_id", "posteam"], how="left")
        .join(rush_stats, on=["game_id", "posteam"], how="left")
        .join(redzone_stats, on=["game_id", "posteam"], how="left")
    )

    return result

if __name__ == "__main__":
    raw = load_data_s3("sports-betting-ml", "raw-data/2020_pbp_data.parquet")
    processed = compute_team_game_features(raw)
    import pdb; pdb.set_trace()