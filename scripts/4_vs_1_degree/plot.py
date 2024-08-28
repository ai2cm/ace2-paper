import dataclasses
from typing import Dict, List
import numpy as np
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
from cartopy import crs as ccrs
import matplotlib as mpl

TRANSFORM = ccrs.PlateCarree()
PROJECTION = ccrs.Robinson(central_longitude=180)

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
        biases = {}
        bias_limits = {}
        rmses = {}
        long_names = {}
        for var in comparison.variables:
            long_names[var.name] = var.long_name
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

            fig, ax = plt.subplots(4, 1, figsize=(10, 20), subplot_kw={"projection": PROJECTION})
            target_1deg_coarse.plot(ax=ax[2], vmin=vmin, vmax=vmax, transform=TRANSFORM)
            var_1deg.plot(ax=ax[3], vmin=vmin, vmax=vmax, transform=TRANSFORM)
            var_4deg_ref.plot(ax=ax[0], vmin=vmin, vmax=vmax, transform=TRANSFORM)
            var_4deg.plot(ax=ax[1], vmin=vmin, vmax=vmax, transform=TRANSFORM)
            ax[2].set_title("1deg Target")
            ax[3].set_title("1deg Predicted")
            ax[0].set_title("4deg Target")
            ax[1].set_title("4deg Predicted")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-time_mean_map.png")
            plt.close(fig)

            err_1deg = var_1deg - target_1deg_coarse
            err_4deg = var_4deg - var_4deg_ref
            biases[var.name] = (err_4deg, err_1deg)
            vmin = min(err_1deg.min().item(), err_4deg.min().item())
            vmax = max(err_1deg.max().item(), err_4deg.max().item())
            vmin = min(vmin, -vmax)
            vmax = -vmin
            bias_limits[var.name] = (vmin, vmax)
            fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), subplot_kw={"projection": PROJECTION})
            c = (err_1deg).plot(ax=ax[1], vmin=vmin, vmax=vmax, transform=TRANSFORM, cmap="RdBu", add_colorbar=False)
            (err_4deg).plot(ax=ax[0], vmin=vmin, vmax=vmax, transform=TRANSFORM, cmap="RdBu", add_colorbar=False)
            rmse_1deg = metrics.weighted_std(
                torch.as_tensor(err_1deg.values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            rmse_4deg = metrics.weighted_std(
                torch.as_tensor(err_4deg.values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            cbar = fig.colorbar(c, ax=ax, orientation='vertical', fraction=0.05, pad=0)
            cbar.set_label(f"{var.long_name}\ntime-mean bias ({var.units})")
            ax[1].set_title("1-degree ensemble mean\nRMSE: {:.2f}".format(rmse_1deg))
            ax[0].set_title("4-degree ensemble mean\nRMSE: {:.2f}".format(rmse_4deg))
            for iax in ax:
                iax.coastlines(color="gray")
            plt.tight_layout()
            fig.subplots_adjust(right=0.83)
            fig.savefig(comparison_out_path / f"{var.name}-time_mean_bias_map.png")
            plt.close(fig)
            
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
            rmses[var.name] = (rmse_4deg, rmse_1deg)
        # combined bias plot
        fig, ax = plt.subplots(len(biases), 2, figsize=(8, 2.5*len(biases)), subplot_kw={"projection": PROJECTION})
        rmses = []
        for i, var_name in enumerate(biases):
            vmin, vmax = bias_limits[var_name]
            c = (biases[var_name][0]).plot(
                ax=ax[i, 0], vmin=vmin, vmax=vmax, cmap="RdBu", add_colorbar=False, transform=TRANSFORM
            )
            (biases[var_name][1]).plot(
                ax=ax[i, 1], vmin=vmin, vmax=vmax, cmap="RdBu", add_colorbar=False, transform=TRANSFORM
            )
            rmse_4deg = metrics.weighted_std(
                torch.as_tensor(biases[var_name][0].values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            rmse_1deg = metrics.weighted_std(
                torch.as_tensor(biases[var_name][1].values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy()
            long_name = long_names[var_name]
            fig.colorbar(
                c,
                ax=ax[i],
                orientation='vertical',
                fraction=0.05,
                pad=0,
                label=f"{long_name}\ntime-mean bias ({comparison.variables[i].units})"
            )
            ax[i, 0].coastlines(color="gray")
            ax[i, 1].coastlines(color="gray")
            ax[i, 0].set_title("")
            ax[i, 1].set_title("")
            ax[i, 1].set_ylabel("")
            ax[i, 1].set_yticklabels([])
            if i < len(biases) - 1:
                ax[i, 0].set_xticklabels([])
                ax[i, 1].set_xticklabels([])
                ax[i, 0].set_xlabel("")
                ax[i, 1].set_xlabel("")
            ax[i, 0].set_ylabel("latitude")
            if i == 0:
                ax[i, 0].set_title(
                    "4-degree ensemble mean\nRMSE: {:.2f} {}".format(
                        rmse_4deg, comparison.variables[i].units
                        )
                    )
                ax[i, 1].set_title(
                    "1-degree ensemble mean\nRMSE: {:.2f} {}".format(
                        rmse_1deg, comparison.variables[i].units
                        )
                    )
            else:
                ax[i, 0].set_title("RMSE: {:.2f} {}".format(rmse_4deg, comparison.variables[i].units))
                ax[i, 1].set_title("RMSE: {:.2f} {}".format(rmse_1deg, comparison.variables[i].units))
        ax[-1, 0].set_xlabel("longitude")
        ax[-1, 1].set_xlabel("longitude")
        plt.tight_layout()
        fig.subplots_adjust(right=0.83)
        fig.savefig(comparison_out_path / "time_mean_bias_map.png")
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
        print(f"{comparison.long_name}:")
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
        area_4deg = get_area(ds_4deg.lat, ds_4deg.lon)
        ds_1deg_coarse = (ds_1deg * area_1deg).coarsen(lat=4, lon=4).mean() / (area_1deg).coarsen(lat=4, lon=4).mean()
        for var in comparison.variables:
            data_4deg = ds_4deg[var.name].sel(source="prediction") * var.scale
            data_4deg_ref = ds_4deg[var.name].sel(source="target") * var.scale
            data_1deg = ds_1deg_coarse[var.name].sel(source="prediction") * var.scale
            data_1deg_ref = ds_1deg_coarse[var.name].sel(source="target") * var.scale
            vmin = min(
                data_4deg.min().item(),
                data_4deg_ref.min().item(),
                data_1deg.min().item(),
                data_1deg_ref.min().item(),
            )
            vmax = max(
                data_4deg.max().item(),
                data_4deg_ref.max().item(),
                data_1deg.max().item(),
                data_1deg_ref.max().item(),
            )
            fig, ax = plt.subplots(2, 2, figsize=(10, 5), subplot_kw={"projection": PROJECTION})
            data_4deg.plot(ax=ax[0, 0], vmin=vmin, vmax=vmax, cmap="RdBu", transform=TRANSFORM)
            data_4deg_ref.plot(ax=ax[0, 1], vmin=vmin, vmax=vmax, cmap="RdBu", transform=TRANSFORM)
            data_1deg.plot(ax=ax[1, 0], vmin=vmin, vmax=vmax, cmap="RdBu", transform=TRANSFORM)
            data_1deg_ref.plot(ax=ax[1, 1], vmin=vmin, vmax=vmax, cmap="RdBu", transform=TRANSFORM)
            ax[0, 0].set_title("4deg")
            ax[0, 1].set_title("4deg ref")
            ax[1, 0].set_title("1deg")
            ax[1, 1].set_title("1deg ref")
            for iax in ax.flatten():
                iax.coastlines(color="gray")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-enso_coefficient_map.png")
            plt.close(fig)

            err_4deg = data_4deg - data_4deg_ref
            err_1deg = data_1deg - data_1deg_ref
            vmin = min(err_4deg.min().item(), err_1deg.min().item())
            vmax = max(err_4deg.max().item(), err_1deg.max().item())
            vmin = min(vmin, -vmax)
            vmax = -vmin

            # R2 = 1 - MSE / Var
            mse_4deg = metrics.weighted_std(
                torch.as_tensor(err_4deg.values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy() ** 2
            mse_1deg = metrics.weighted_std(
                torch.as_tensor(err_1deg.values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy() ** 2
            var_4deg = metrics.weighted_std(
                torch.as_tensor(data_4deg.values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy() ** 2
            var_1deg = metrics.weighted_std(
                torch.as_tensor(data_1deg.values),
                weights=torch.as_tensor(area_4deg.values),
            ).cpu().numpy() ** 2
            r2_4deg = 1 - mse_4deg / var_4deg
            r2_1deg = 1 - mse_1deg / var_1deg
            print(f"{var.long_name}: R2 4deg: {r2_4deg}, R2 1deg: {r2_1deg}")

            fig, ax = plt.subplots(1, 2, figsize=(8, 2.5), subplot_kw={"projection": PROJECTION})
            c = (err_4deg).plot(
                ax=ax[0], vmin=vmin, vmax=vmax, cmap="RdBu", add_colorbar=False, transform=TRANSFORM
            )
            (err_1deg).plot(
                ax=ax[1], vmin=vmin, vmax=vmax, cmap="RdBu", add_colorbar=False, transform=TRANSFORM
            )
            for iax in ax:
                iax.coastlines(color="gray")
            cbar = fig.colorbar(c, ax=ax, orientation='vertical', fraction=0.05, pad=0)
            cbar.set_label(f"{var.long_name}\nENSO coefficient bias ({var.units})")
            ax[0].set_title("4-degree ensemble mean\nR2: {:.2f}".format(r2_4deg))
            ax[1].set_title("1-degree ensemble mean\nR2: {:.2f}".format(r2_1deg))
            for i in (0, 1):
                ax[i].set_xlabel("longitude")
                ax[i].set_ylabel("latitude")
            ax[1].set_ylabel("")
            ax[1].set_yticklabels([])
            plt.tight_layout()
            fig.subplots_adjust(right=0.83)
            fig.savefig(comparison_out_path / f"{var.name}-enso_coefficient_bias_map.png")
            plt.close(fig)


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
        res_4deg = comparison.res_4deg
        res_1deg = comparison.res_1deg
        comparison_out_path = OUT_PATH / comparison.name
        if not comparison_out_path.exists():
            comparison_out_path.mkdir()
        ds_mean_4deg = xr.concat(
            [annual_means[run.job_name] for run in comparison.res_4deg.runs],
            dim="run",
        ).mean(dim="run")
        ds_mean_1deg = xr.concat(
            [annual_means[run.job_name] for run in comparison.res_1deg.runs],
            dim="run",
        ).mean(dim="run")
        ds_ref_4deg = dataset_cache.open_beaker_dataset(
            res_4deg.reference_run.job_name, "annual_diagnostics.nc"
        )
        ds_ref_1deg = dataset_cache.open_beaker_dataset(
            res_1deg.reference_run.job_name, "annual_diagnostics.nc"
        )
        for var in comparison.variables:
            var_4deg_ref_0 = ds_ref_4deg[var.name].sel(source="prediction") * var.scale
            var_4deg_ref_1 = ds_ref_4deg[var.name].sel(source="target") * var.scale
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(8, 4),
            )
            for i, run in enumerate(res_4deg.runs):
                var_4deg_run = annual_means[run.job_name][var.name].sel(source="prediction") * var.scale
                label = "4-degree ensemble member" if i == 0 else None
                var_4deg_run.plot(ax=ax, alpha=0.5, label=label, color='#1f77b4', linestyle="--")
            var_4deg = ds_mean_4deg[var.name].sel(source="prediction") * var.scale
            var_4deg.plot(ax=ax, label="4-degree ensemble mean", color='#1f77b4')
            
            for i, run in enumerate(res_1deg.runs):
                var_1deg_run = annual_means[run.job_name][var.name].sel(source="prediction") * var.scale
                label = "1-degree ensemble member" if i == 0 else None
                var_1deg_run.plot(ax=ax, alpha=0.5, label=label, color='#ff7f0e', linestyle="--")
            var_1deg = ds_mean_1deg[var.name].sel(source="prediction") * var.scale
            var_1deg.plot(ax=ax, label="1-degree ensemble mean", color='#ff7f0e')
            # var_1deg_ref.plot(ax=ax[0], label="target data", color="gray")
            for i, target_4deg in enumerate((var_4deg_ref_0, var_4deg_ref_1)):
                label = "target member" if i == 0 else None
                target_4deg.plot(ax=ax, label=label, color="gray", linestyle="--")
            ax.set_title("Annual mean series")
            ax.legend()
            ax.set_ylabel(f"mean {var.long_name} ({var.units})")
            xmin, xmax = ds_ref_4deg.year.min().item(), ds_ref_4deg.year.max().item()
            ax.set_xlim(xmin, xmax)
            var_4deg_ref = ds_ref_4deg[var.name].mean("source") * var.scale
            var_1deg_ref = ds_ref_1deg[var.name].mean("source") * var.scale
            bias_4deg = var_4deg - var_4deg_ref
            bias_1deg = var_1deg - var_1deg_ref
            # (bias_4deg).plot(ax=ax[1], label="4-degree ensemble mean bias")
            # (bias_1deg).plot(ax=ax[1], label="1-degree ensemble mean bias")
            # ax[1].hlines(0, xmin, xmax, color="gray")
            # ax[1].set_title("Annual mean biases")
            # ax[1].legend()
            # ax[1].set_ylabel(f"mean bias ({var.units})")
            # ax[1].set_xlim(xmin, xmax)
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-annual_mean_series.png")
            plt.close(fig)

            # plot bias as a histogram over 1-year changes in reference
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            var_4deg_ref_1yr_change = var_4deg_ref.diff("year")
            var_1deg_ref_1yr_change = var_1deg_ref.diff("year")
            ax[0].plot(
                var_4deg_ref_1yr_change.values.flatten(),
                bias_4deg.isel(year=slice(1, None)).values.flatten(),
                marker="x",
                linestyle="",
            )
            ax[1].plot(
                var_1deg_ref_1yr_change.values.flatten(),
                bias_1deg.isel(year=slice(1, None)).values.flatten(),
                marker="x",
                linestyle="",
            )
            ax[0].set_title(f"4-degree ensemble mean {var.long_name}")
            ax[1].set_title(f"1-degree ensemble mean {var.long_name}")
            ax[0].set_xlabel("1-year change in reference")
            ax[1].set_xlabel("1-year change in reference")
            ax[0].set_ylabel("mean bias")
            ax[1].set_ylabel("mean bias")
            plt.tight_layout()
            fig.savefig(comparison_out_path / f"{var.name}-annual_mean_bias_vs_1yr_change.png")
            plt.close(fig)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    mpl.rcParams['figure.dpi'] = args.dpi
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(config)
    config = dacite.from_dict(Config, config, config=dacite.Config(strict=True))
    dataset_cache = DatasetCache(Beaker.from_env())

    plot_time_means(config, dataset_cache)
    plot_enso_coefficients(config, dataset_cache)
    plot_annual_means(config, dataset_cache)
