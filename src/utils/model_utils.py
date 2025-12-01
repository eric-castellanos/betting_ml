from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl


def save_feature_importance_plot(
    feature_importance_df: pl.DataFrame,
    output_path: str,
    top_n: Optional[int] = None,
) -> None:
    """
    Save a bar plot of feature importance values sorted in descending order.

    Args:
        feature_importance_df: Polars DataFrame with columns ["feature", "importance"].
        output_path: Destination for the PNG plot.
        top_n: If provided, limit plot to the top N features.
    """
    if feature_importance_df.is_empty():
        return

    plot_df = feature_importance_df.sort("importance", descending=True)
    if top_n is not None:
        plot_df = plot_df.head(top_n)

    df_pd = plot_df.to_pandas()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df_pd["feature"], df_pd["importance"], color="#4c6ef5")
    plt.xlabel("Gain (importance)")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
