import dataclasses
from typing import Dict, List
import torch
import yaml
import dacite
import argparse
from pathlib import Path
from beaker import Beaker
import subprocess
import xarray as xr
import logging
from fme.core import metrics
import matplotlib.pyplot as plt

DATA_PATH = Path("./data")
OUT_PATH = Path("./output")

@dataclasses.dataclass
class Run:
    job_name: str


@dataclasses.dataclass
class Ensemble:
    reference_run: Run
    runs: List[Run]

@dataclasses.dataclass
class EvaluatedVariable:
    name: str
    scale: float
    long_name: str
    units: str

@dataclasses.dataclass
class Comparison:
    name: str
    long_name: str
    variables: List[EvaluatedVariable]
    res_4deg: Ensemble
    res_1deg: Ensemble

@dataclasses.dataclass
class Config:
    comparisons: List[Comparison]

class DatasetCache:
    DATA_PATH = Path("./data")

    def __init__(self, beaker: Beaker):
        self.beaker = beaker
        self._result_dataset_names: Dict[str, str] = {}
    
    def open_beaker_dataset(self, job_name: str, path: str) -> xr.Dataset:
        dataset = self._get_result_dataset_name(job_name)
        fetched_path = self._get_fetched_path(dataset, path)
        if not fetched_path.exists():
            self._fetch_beaker_dataset(dataset, path)
        return xr.open_dataset(fetched_path)

    def _get_fetched_path(self, dataset: str, path: str) -> Path:
        return self.DATA_PATH / dataset / path
    
    def _get_result_dataset_name(self, experiment_name: str):
        if experiment_name not in self._result_dataset_names:
            experiment = self.beaker.experiment.get(experiment_name)
            self._result_dataset_names[experiment_name] = experiment.jobs[-1].result.beaker
        return self._result_dataset_names[experiment_name]

    def _fetch_beaker_dataset(self, dataset: str, path: str) -> xr.Dataset:
        logging.info(f"Fetching dataset {dataset} to {path}")
        subprocess.run(
            [
                "beaker",
                "dataset",
                "fetch",
                dataset,
                "--prefix",
                path,
                "--output",
                self.DATA_PATH / dataset,
            ],
            check=True,
        )

def get_area(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    area = xr.DataArray(
        metrics.spherical_area_weights(lat.values, len(lon)),
        dims=["lat", "lon"],
    )
    return area


def convert_to_pred_and_target(ds: xr.Dataset) -> xr.Dataset:
    """
    Take a dataset whose variable names are prefixed with "gen_map-" and "bias_map-",
    and produce a dataset with a "source" dimension with values "pred" and "target.
    """
    pred_data = {}
    target_data = {}
    for var_name in ds.data_vars:
        if var_name.startswith("gen_map-"):
            pred_data[var_name[8:]] = ds[var_name]
            target_data[var_name[8:]] = ds[var_name] + ds[f"bias_map-{var_name[8:]}"]
    pred_ds = xr.Dataset(pred_data)
    target_ds = xr.Dataset(target_data)
    return xr.concat([pred_ds, target_ds], dim="source").assign_coords(source=["pred", "target"])


def plot_time_means(config: Config, dataset_cache: DatasetCache):
    time_means = {}
    for comparison in config.comparisons:
        res_4deg = comparison.res_4deg
        res_1deg = comparison.res_1deg
        for run in res_4deg.runs:
            inference_time_mean = dataset_cache.open_beaker_dataset(
                run.job_name, "time_mean_diagnostics.nc"
            )
            time_means[run.job_name] = inference_time_mean
        for run in res_1deg.runs:
            inference_time_mean = dataset_cache.open_beaker_dataset(
                run.job_name, "time_mean_diagnostics.nc"
            )
            time_means[run.job_name] = inference_time_mean
    if not OUT_PATH.exists():
        OUT_PATH.mkdir()
    
    for comparison in config.comparisons:
        comparison_out_path = OUT_PATH / comparison.name
        if not comparison_out_path.exists():
            comparison_out_path.mkdir()
        ds_4deg = xr.concat(
            [time_means[run.job_name] for run in comparison.res_4deg.runs],
            dim="run",
        ).mean(dim="run")
        ds_1deg = xr.concat(
            [time_means[run.job_name] for run in comparison.res_1deg.runs],
            dim="run",
        ).mean(dim="run")
        ds_4deg = convert_to_pred_and_target(ds_4deg)
        area_4deg = get_area(ds_4deg.lat, ds_4deg.lon)
        ds_1deg = convert_to_pred_and_target(ds_1deg)
        area_1deg = get_area(ds_1deg.lat, ds_1deg.lon)
        ds_1deg_coarse = (ds_1deg * area_1deg).coarsen(lat=4, lon=4).mean() / (area_1deg).coarsen(lat=4, lon=4).mean()
        for var in comparison.variables:
            target_1deg_coarse = ds_1deg_coarse.sel(source="target")[var.name] * var.scale
            var_4deg = ds_4deg.sel(source="pred")[var.name].assign_coords(coords=target_1deg_coarse.coords) * var.scale
            var_1deg = ds_1deg_coarse.sel(source="pred")[var.name].assign_coords(coords=target_1deg_coarse.coords) * var.scale
            var_4deg_ref = ds_4deg.sel(source="target")[var.name].assign_coords(coords=target_1deg_coarse.coords) * var.scale
            vmin = min(
                target_1deg_coarse.min().item(),
                var_1deg.min().item(),
                var_4deg.min().item(),
                var_4deg_ref.min().item(),
            )
            vmax = max(
                target_1deg_coarse.max().item(),
                var_1deg.max().item(),
                var_4deg.max().item(),
                var_4deg_ref.max().item(),
            )

            fig, ax = plt.subplots(4, 1, figsize=(10, 20))
            target_1deg_coarse.plot(ax=ax[0], vmin=vmin, vmax=vmax)
            var_1deg.plot(ax=ax[1], vmin=vmin, vmax=vmax)
            var_4deg_ref.plot(ax=ax[2], vmin=vmin, vmax=vmax)
            var_4deg.plot(ax=ax[3], vmin=vmin, vmax=vmax)
            ax[0].set_title("1deg Target")
            ax[1].set_title("1deg Predicted")
            ax[2].set_title("4deg Target")
            ax[3].set_title("4deg Predicted")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-time_mean_map.png")

            err_1deg = var_1deg - target_1deg_coarse
            err_4deg = var_4deg - var_4deg_ref
            vmin = min(err_1deg.min().item(), err_4deg.min().item())
            vmax = max(err_1deg.max().item(), err_4deg.max().item())
            vmin = min(vmin, -vmax)
            vmax = -vmin
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            (err_1deg).plot(ax=ax[0], vmin=vmin, vmax=vmax)
            (err_4deg).plot(ax=ax[1], vmin=vmin, vmax=vmax)
            ax[0].set_title("1deg time-mean bias")
            ax[1].set_title("4deg time-mean bias")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-time_mean_bias_map.png")
            
            rmse_1deg = metrics.weighted_std(
                torch.as_tensor((var_1deg.values - target_1deg_coarse.values)),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            rmse_4deg = metrics.weighted_std(
                torch.as_tensor((var_4deg.values - var_4deg_ref.values)),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            rmse_4deg_reference = metrics.weighted_std(
                torch.as_tensor((var_4deg_ref - target_1deg_coarse).values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            rmse_4deg_vs_1deg_reference = metrics.weighted_std(
                torch.as_tensor((var_4deg - target_1deg_coarse).values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            fig, ax = plt.subplots()
            ax.bar(["1deg", "4deg", "4deg_ref", "4deg_vs_1deg_coarse"], [rmse_1deg, rmse_4deg, rmse_4deg_reference, rmse_4deg_vs_1deg_reference])
            ax.set_ylabel(f"Time-mean RMSE of {var.long_name} ({var.units})")
            fig.savefig(comparison_out_path / f"{var.name}-time_mean_rmse.png")
            plt.close(fig)


def plot_enso_coefficients(config: Config, dataset_cache: DatasetCache):
    enso_coefficients = {}
    for comparison in config.comparisons:
        res_4deg = comparison.res_4deg
        res_1deg = comparison.res_1deg
        for run in res_4deg.runs:
            ds_enso = dataset_cache.open_beaker_dataset(
                run.job_name, "enso_coefficient_diagnostics.nc"
            )
            enso_coefficients[run.job_name] = ds_enso
        for run in res_1deg.runs:
            ds_enso = dataset_cache.open_beaker_dataset(
                run.job_name, "enso_coefficient_diagnostics.nc"
            )
            enso_coefficients[run.job_name] = ds_enso

    if not OUT_PATH.exists():
        OUT_PATH.mkdir()
    
    for comparison in config.comparisons:
        comparison_out_path = OUT_PATH / comparison.name
        if not comparison_out_path.exists():
            comparison_out_path.mkdir()
        ds_4deg = xr.concat(
            [enso_coefficients[run.job_name] for run in comparison.res_4deg.runs],
            dim="run",
        ).mean(dim="run")
        ds_1deg = xr.concat(
            [enso_coefficients[run.job_name] for run in comparison.res_1deg.runs],
            dim="run",
        ).mean(dim="run")
        area_1deg = get_area(ds_1deg.lat, ds_1deg.lon)
        ds_1deg_coarse = (ds_1deg * area_1deg).coarsen(lat=4, lon=4).mean() / (area_1deg).coarsen(lat=4, lon=4).mean()
        for var in comparison.variables:
            var_4deg = ds_4deg[var.name].sel(source="prediction") * var.scale
            var_4deg_ref = ds_4deg[var.name].sel(source="target") * var.scale
            var_1deg = ds_1deg_coarse[var.name].sel(source="prediction") * var.scale
            var_1deg_ref = ds_1deg_coarse[var.name].sel(source="target") * var.scale
            vmin = min(
                var_4deg.min().item(),
                var_4deg_ref.min().item(),
                var_1deg.min().item(),
                var_1deg_ref.min().item(),
            )
            vmax = max(
                var_4deg.max().item(),
                var_4deg_ref.max().item(),
                var_1deg.max().item(),
                var_1deg_ref.max().item(),
            )
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            var_4deg.plot(ax=ax[0, 0], vmin=vmin, vmax=vmax)
            var_4deg_ref.plot(ax=ax[0, 1], vmin=vmin, vmax=vmax)
            var_1deg.plot(ax=ax[1, 0], vmin=vmin, vmax=vmax)
            var_1deg_ref.plot(ax=ax[1, 1], vmin=vmin, vmax=vmax)
            ax[0, 0].set_title("4deg")
            ax[0, 1].set_title("4deg ref")
            ax[1, 0].set_title("1deg")
            ax[1, 1].set_title("1deg ref")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-enso_coefficient_map.png")

            err_4deg = var_4deg - var_4deg_ref
            err_1deg = var_1deg - var_1deg_ref
            vmin = min(err_4deg.min().item(), err_1deg.min().item())
            vmax = max(err_4deg.max().item(), err_1deg.max().item())
            vmin = min(vmin, -vmax)
            vmax = -vmin
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            (err_4deg).plot(ax=ax[0], vmin=vmin, vmax=vmax)
            (err_1deg).plot(ax=ax[1], vmin=vmin, vmax=vmax)
            ax[0].set_title("4deg bias")
            ax[1].set_title("1deg bias")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-enso_coefficient_bias_map.png")


def plot_annual_means(config: Config, dataset_cache: DatasetCache):
    annual_means = {}
    for comparison in config.comparisons:
        res_4deg = comparison.res_4deg
        res_1deg = comparison.res_1deg
        for run in res_4deg.runs:
            ds_enso = dataset_cache.open_beaker_dataset(
                run.job_name, "annual_diagnostics.nc"
            )
            annual_means[run.job_name] = ds_enso
        for run in res_1deg.runs:
            ds_enso = dataset_cache.open_beaker_dataset(
                run.job_name, "annual_diagnostics.nc"
            )
            annual_means[run.job_name] = ds_enso

    if not OUT_PATH.exists():
        OUT_PATH.mkdir()
    
    for comparison in config.comparisons:
        comparison_out_path = OUT_PATH / comparison.name
        if not comparison_out_path.exists():
            comparison_out_path.mkdir()
        ds_4deg = xr.concat(
            [annual_means[run.job_name] for run in comparison.res_4deg.runs],
            dim="run",
        ).mean(dim="run")
        ds_1deg = xr.concat(
            [annual_means[run.job_name] for run in comparison.res_1deg.runs],
            dim="run",
        ).mean(dim="run")
        for var in comparison.variables:
            var_4deg = ds_4deg[var.name].sel(source="prediction") * var.scale
            var_4deg_ref = ds_4deg[var.name].sel(source="target") * var.scale
            var_1deg = ds_1deg[var.name].sel(source="prediction") * var.scale
            var_1deg_ref = ds_1deg[var.name].sel(source="target") * var.scale
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            var_4deg.plot(ax=ax[0], label="4deg")
            var_4deg_ref.plot(ax=ax[0], label="4deg ref")
            var_1deg.plot(ax=ax[0], label="1deg")
            var_1deg_ref.plot(ax=ax[0], label="1deg ref")
            ax[0].set_title("annual means")
            ax[0].legend()
            (var_4deg - var_4deg_ref).plot(ax=ax[1], label="4deg bias")
            (var_1deg - var_1deg_ref).plot(ax=ax[1], label="1deg bias")
            ax[1].set_title("annual mean biases")
            ax[1].legend()
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-annual_mean_series.png")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    config = dacite.from_dict(Config, config, config=dacite.Config(strict=True))
    dataset_cache = DatasetCache(Beaker.from_env())

    plot_time_means(config, dataset_cache)
    plot_enso_coefficients(config, dataset_cache)
    plot_annual_means(config, dataset_cache)
