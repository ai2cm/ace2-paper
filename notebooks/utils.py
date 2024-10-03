import beaker
import wandb
import xarray as xr
import io
import datetime
import numpy as np
from typing import Optional, Sequence, Tuple


def wandb_to_beaker_experiment(project: str, id: str, entity: str = "ai2cm") -> str:
    """Given a wandb run ID, return corresponding beaker experiment ID"""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{id}")
    return run.config["environment"]["BEAKER_EXPERIMENT_ID"]


def beaker_experiment_to_wandb(beaker_experiment: str) -> Optional[Tuple[str, str, str]]:
    """Given a beaker experiment ID or name, return corresponding wandb entity, project, and run ID.
    
    Warning: this function is not reliable, since the necessary line could be split between
    pages. Will return None if can't get the ID.
    """
    client = beaker.Beaker.from_env()
    for page in client.experiment.logs(beaker_experiment, quiet=True):
        page_str = page.decode("utf-8")
        lines = page_str.split("\n")
        for line in lines:
            if "View run at https://wandb.ai" in line and "runs/" in line:
                elements = line.split("/")
                id_ = elements[-1]
                project = elements[-3]
                entity = elements[-4]
                return entity, project, id_
    return None


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
    return xr.open_dataset(io.BytesIO(file), engine="h5netcdf").load()


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