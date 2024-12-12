import beaker
import wandb
import xarray as xr
import io
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Sequence, List, Tuple

FIGURE_DIR = './figures'
DPI = 300
FONTSIZE = 8


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
        try:
            page_str = page.decode("utf-8")
        except UnicodeDecodeError:
            return None
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
    return xr.open_dataset(io.BytesIO(file), engine='h5netcdf').load()


def get_wandb_scalar_metrics(
    run,
    metric_names: List[str]
):
    """
    Given a wandb run and list of scalar metric names, return a dict of those
    metric values.
    
    Note: Assumes scalar metrics occur once in run history, so that they can be
    reliably accessed by the wandb run `summary`.
    """
    metrics = {}
    summary = run.summary
    for key in metric_names:
        metric = summary.get(key)
        if metric is not None:
            metrics[key] = metric
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


def savefig(
    fig: plt.Figure,
    name: str,
    figure_dir: str=FIGURE_DIR,
    bbox_inches: str='tight',
    dpi: int=DPI,
    transparent: bool=True,
    **savefig_kwargs
):
    """
    """
    savefig_kwargs.update(
        {
            "bbox_inches": bbox_inches,
            "transparent": transparent,
            "dpi": dpi,
        }
    )
    full_path = os.path.join(figure_dir, name)
    fig.savefig(full_path, **savefig_kwargs)