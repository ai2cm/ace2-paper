import beaker
import wandb
import xarray as xr
import io
import datetime
import numpy as np
from typing import Optional, Sequence, List


def wandb_to_beaker_experiment(project: str, id: str, entity: str = "ai2cm") -> str:
    """Given a wandb run ID, return corresponding beaker experiment ID"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{id}")
    return run.config["environment"]["BEAKER_EXPERIMENT_ID"]


def wandb_to_beaker_result(project: str, id: str, entity: str = "ai2cm") -> str:
    """Given a wandb run ID, return ID of corresponding beaker result dataset"""
    experiment_id = wandb_to_beaker_experiment(project, id, entity=entity)
    client = beaker.Beaker.from_env()
    result_dataset = client.experiment.results(experiment_id)
    return result_dataset.id


def beaker_to_xarray(dataset_id: str, path: str) -> xr.Dataset:
    """Given a beaker dataset ID and path within dataset, return an xarray dataset.

    Note: dataset must fit in memory. Requires h5netcdf backend.
    """
    client = beaker.Beaker.from_env()
    file = client.dataset.get_file(dataset_id, path)
    return xr.open_dataset(io.BytesIO(file), engine='h5netcdf').load()


def get_scalar_metrics(run: wandb.apis.public.runs.Run, metric_names: List[str]):
    """
    Given a wandb run ID and list of scalar metric names, 
    return a dict of those metric values.
    
    Note: Assumes metrics occurs once in run history,
    and each metric occurs at the same step.
    """
    metrics = {}
    history = run.scan_history(keys=metric_names)
    for key in metric_names:
        metrics[key] = [row for row in history][0][key]
    return metrics


def get_beaker_dataset_variables(
    wandb_id: str, 
    ds_name: str,
    varnames: List[str],
    wandb_project: str="ace",
    wandb_entity: str="ai2cm",
):
    """Given a wandb run ID, a results dataset name, and variable names,
    return an xarray dataset of those variables.
    """
    beaker_result_id = wandb_to_beaker_result(wandb_project, wandb_id, wandb_entity)
    diags = beaker_to_xarray(beaker_result_id, ds_name)
    valid_varnames = [name for name in varnames if name in diags.data_vars]
    return diags[valid_varnames]


def wandb_to_xarray(
    project: str,
    id: str,
    metric_names: Sequence[str],
    samples: int,
    entity: str = "ai2cm",
    add_time_coord: bool = True,
) -> xr.Dataset:
    """Given a wandb run ID and metric names, return an xarray dataset.
    
    Args:
        project: wandb project name
        id: wandb run ID
        metric_names: list of metric names to fetch
        samples: number of steps to fetch
        entity: wandb entity name
        add_time_coord: add a time coordinate to the dataset
        
    Returns:
        xarray.Dataset of the desired metrics.
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{id}")
    metrics = run.history(keys=metric_names, samples=samples)
    if add_time_coord:
        metrics["lead_time"] = metrics._step / 4
        metrics = metrics.set_index("lead_time")
    ds = xr.Dataset.from_dataframe(metrics)
    if add_time_coord:
        ds['lead_time'].attrs["units"] = "days since init"
        del ds["_step"]
    return ds
