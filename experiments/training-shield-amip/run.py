# requires beaker-py, install with
# pip install -U beaker-py

import beaker
import uuid
from typing import Dict, Any
import tempfile
import yaml
import os
import fme
import dacite

IMAGE_NAME = "brianhenn/fme-c7a51eff"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"
STATS_DATASET_1DEG_NAME = "andrep/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset-stats"
STATS_DATASET_4DEG_NAME = "andrep/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset-stats"
REFERENCE_DATASET_1DEG_NAME = "brianhenn/2024-07-24-vertically-resolved-c96-1deg-shield-amip-monthly-reference"
REFERENCE_DATASET_4DEG_NAME = "brianhenn/2024-07-24-vertically-resolved-c96-4deg-shield-amip-monthly-reference"

AMIP_4DEG_OVERLAY = {
    "train_loader": {
        "dataset": [
            {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0001",
                "subset": {"stop_time": "1995-12-31"},
            },
            {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0002",
                "subset": {"stop_time": "1995-12-31"},
            },
            {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0001",
                "subset": {"start_time": "2011-01-01"},
            },
            {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0002",
                "subset": {"start_time": "2011-01-01"},
            },
        ]
    },
    "validation_loader": {
        "dataset": [
            {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0001",
                "subset": {"start_time": "1996-01-01", "stop_time": "2000-12-31"},
            },
            {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0002",
                "subset": {"start_time": "1996-01-01", "stop_time": "2000-12-31"},
            },
        ]
    },
    "inference": {
        "loader": {
            "dataset": {
                "data_path": "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0002",
            },
        },
        "aggregator": {
            "monthly_reference_data": "/refdata-4deg/monthly_mean_data.nc",
        }
    },
    "stepper": {
        "normalization": {
            "global_means_path": "/statsdata-4deg/centering.nc",
            "global_stds_path": "/statsdata-4deg/scaling-full-field.nc",
        },
        "residual_normalization": {
            "global_means_path": "/statsdata-4deg/centering.nc",
            "global_stds_path": "/statsdata-4deg/scaling-residual.nc",
        }
    },
}

AMIP_4DEG_768_CHANNEL_OVERLAY = {
    **AMIP_4DEG_OVERLAY,
    "stepper": {
        "normalization": {
            "global_means_path": "/statsdata-4deg/centering.nc",
            "global_stds_path": "/statsdata-4deg/scaling-full-field.nc",
        },
        "residual_normalization": {
            "global_means_path": "/statsdata-4deg/centering.nc",
            "global_stds_path": "/statsdata-4deg/scaling-residual.nc",
        },
        "builder": {
            "config": {
                "embed_dim": 768
            }
        }
    }
}

# experiments defined by overlays which will overwrite the keys of the base config
EXPERIMENT_OVERLAYS = {
    "shield-amip-1deg-ace2-training-rs0": {},
    "shield-amip-1deg-ace2-training-rs1": {},
    "shield-amip-1deg-ace2-training-rs2": {},
    "shield-amip-1deg-ace2-training-no-mois-cons-rs0": {"stepper": {"corrector": {"moisture_budget_correction": None}}},
    "shield-amip-1deg-ace2-training-no-mois-cons-rs1": {"stepper": {"corrector": {"moisture_budget_correction": None}}},
    "shield-amip-4deg-ace2-training-rs0": AMIP_4DEG_OVERLAY,
    "shield-amip-4deg-ace2-training-rs1": AMIP_4DEG_OVERLAY,
    "shield-amip-4deg-ace2-training-rs2": AMIP_4DEG_OVERLAY,
    "shield-amip-4deg-ace2-training-768C-rs0": AMIP_4DEG_OVERLAY,
    "shield-amip-4deg-ace2-training-768C-rs1": AMIP_4DEG_OVERLAY,
    "shield-amip-4deg-ace2-training-768C-rs2": AMIP_4DEG_OVERLAY,
}


def merge_configs(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested configurations."""
    for k, v in new.items():
        if isinstance(v, dict):
            base[k] = merge_configs(base.get(k, {}), v)
        else:
            base[k] = v
    return base

def write_config_dataset(config: Dict[str, Any]):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, DATASET_CONFIG_FILENAME)
        with open(filepath, "w") as f:
            yaml.safe_dump(config, f)
        dataset_name = "ace-training-config-" + str(uuid.uuid4())[:8]
        dataset = client.dataset.create(dataset_name, filepath)
    return dataset


def get_experiment_spec(name: str, config: Dict[str, Any], image_name=IMAGE_NAME):
    """Given a dict representing the training configuration, return a beaker experiment spec."""
    config_dataset = write_config_dataset(config)
    env_vars = [
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key-ai2cm-sa"),
        beaker.EnvVar(name="WANDB_JOB_TYPE", value="training"),
        beaker.EnvVar(name="WANDB_NAME", value=name),
        beaker.EnvVar(name="WANDB_RUN_GROUP", value="shield-amip-ace2-training"),
        beaker.EnvVar(name="WANDB_USERNAME", value="bhenn1983")
    ]
    datasets = [
        beaker.DataMount(
            source=beaker.DataSource(beaker=config_dataset.id),
            mount_path=DATASET_CONFIG_MOUNTPATH,
        ),
        beaker.DataMount(
            mount_path="/climate-default",
            source=beaker.DataSource(weka="climate-default"),
        ),
        beaker.DataMount(
            source=beaker.DataSource(beaker=STATS_DATASET_1DEG_NAME),
            mount_path="/statsdata-1deg",
        ),
        beaker.DataMount(
            source=beaker.DataSource(beaker=STATS_DATASET_4DEG_NAME),
            mount_path="/statsdata-4deg",
        ),
        beaker.DataMount(
            source=beaker.DataSource(beaker=REFERENCE_DATASET_1DEG_NAME),
            mount_path="/refdata-1deg",
        ),
        beaker.DataMount(
            source=beaker.DataSource(beaker=REFERENCE_DATASET_4DEG_NAME),
            mount_path="/refdata-4deg",
        ),
    ]
    spec = beaker.ExperimentSpec(
        budget="ai2/climate",
        description="SHiELD AMIP ACE2 training",
        tasks=[
            beaker.TaskSpec(
                name=name,
                image=beaker.ImageSource(beaker=image_name),
                command=[
                    "torchrun",
                    "--nproc_per_node",
                    "8",
                    "-m",
                    "fme.ace.train",
                    f"{DATASET_CONFIG_MOUNTPATH}/{DATASET_CONFIG_FILENAME}",
                ],
                result=beaker.ResultSpec(path="/output"),
                resources=beaker.TaskResources(gpu_count=8, shared_memory="400GiB"),
                context=beaker.TaskContext(priority="high", preemptible=True),
                constraints=beaker.Constraints(cluster=["ai2/jupiter-cirrascale-2"]),
                env_vars=env_vars,
                datasets=datasets,
            )
        ],
    )
    return spec


if __name__ == "__main__":
    client = beaker.Beaker.from_env()

    with open(LOCAL_BASE_CONFIG_FILENAME, "r") as f:
        base_config = yaml.safe_load(f)

    print("Validating that configs have correct types.")
    for name, overlay in EXPERIMENT_OVERLAYS.items():
        config = merge_configs(base_config, overlay)
        print(f"Validating config for experiment {name}.")
        print(f"Config that is being validated:\n{config}")
        dacite.from_dict(fme.ace.TrainConfig, config, config=dacite.Config(strict=True))
    print("All configs are valid. Starting experiment submission.")
    for name, overlay in EXPERIMENT_OVERLAYS.items():
        config = merge_configs(base_config, overlay)
        print(f"Creating experiment {name}.")
        spec = get_experiment_spec(name, config)
        experiment = client.experiment.create(name, spec, workspace="ai2/ace")
        print(f"Experiment created. See https://beaker.org/ex/{experiment.id}")
