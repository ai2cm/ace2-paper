import beaker
import wandb
import xarray as xr
import io
from typing import List


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
    history = run.scan_history(keys=metric_names)
    metrics = {}
    for key in metric_names:
        metrics[key] = [row for row in history][0][key]
    return metrics

def get_beaker_dataset_variables(
    wandb_id: str, 
    ds_name: str,
    varnames: List[str],
    wandb_project: str="ace",
    wandb_entity: str="ai2cm",
    alternate_wandb_project="ace2"
):
    """Given a wandb run ID, a results dataset name, and variable names,
    return an xarray dataset of those variables.
    """
    try:
        beaker_result_id = wandb_to_beaker_result(wandb_project, wandb_id, wandb_entity)
    except:
        beaker_result_id = wandb_to_beaker_result(alternate_wandb_project, wandb_id, wandb_entity)
    time_mean_diags = beaker_to_xarray(beaker_result_id, ds_name)
    return time_mean_diags[varnames]