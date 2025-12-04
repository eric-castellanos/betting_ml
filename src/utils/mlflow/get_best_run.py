import click
import mlflow


def find_best_run(experiment_name: str, metric: str, exclude_nested: bool):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} ASC"],
    )

    if exclude_nested and "tags.mlflow.parentRunId" in df.columns:
        df = df[df["tags.mlflow.parentRunId"].isna()]

    if df.empty:
        click.echo("No runs found for the given criteria.")
        return None

    best_run = df.iloc[0]
    return best_run


@click.command()
@click.option("--experiment", "-e", required=True, help="MLflow experiment name")
@click.option("--metric", "-m", default="val_mae", show_default=True, help="Metric name to sort by")
@click.option("--exclude-nested", is_flag=True, default=False, help="Exclude Optuna nested runs")
def main(experiment: str, metric: str, exclude_nested: bool) -> None:
    best_run = find_best_run(experiment, metric, exclude_nested)
    if best_run is None:
        return

    click.echo("=== BEST RUN ===")
    click.echo(f"Run ID: {best_run.run_id}")
    metric_value = best_run.get(f"metrics.{metric}", "N/A")
    click.echo(f"Metric ({metric}): {metric_value}")

    params = {k.replace("params.", "", 1): v for k, v in best_run.items() if k.startswith("params.")}

    click.echo("\n=== PARAMETERS ===")
    for k, v in params.items():
        click.echo(f"{k}: {v}")


if __name__ == "__main__":
    main()
