import beaker
import wandb
import xarray as xr
import io

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