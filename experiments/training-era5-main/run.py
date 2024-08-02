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

IMAGE_NAME = "oliverwm/fme-d8961d26"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"
STATS_DATASET_NAME = "oliverwm/era5-1deg-8layer-stats-1990-2019-v2"

ERA5_DATA_PATH = "/climate-default/2024-06-20-era5-1deg-8layer-1940-2022-netcdfs"
SHIELD_DATA_PATH = "/climate-default/2024-07-21-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/train/ic_0001"

# experiments defined by overlays which will overwrite the keys of the base config
EXPERIMENT_OVERLAYS = {
    "era5-ace2-co2-rs0": {},
    "era5-ace2-co2-rs1": {},
    # "era5-ace2-rw4-shield-data": {   # need SHiELD data with CO2 before can launch this
    #    "max_epochs": 75,
    #    "train_loader": {
    #        "batch_size": 16,
    #        "num_data_workers": 32,
    #        "dataset": [
    #            {"data_path": ERA5_DATA_PATH, "subset": {"stop_time": "1995-12-31"}},
    #            {
    #                "data_path": ERA5_DATA_PATH,
    #                "subset": {"start_time": "2011-01-01", "stop_time": "2019-12-31"},
    #            },
    #            {"data_path": ERA5_DATA_PATH, "subset": {"start_time": "2021-01-01"}},
    #            {"data_path": SHIELD_DATA_PATH, "subset": {"stop_time": "1995-12-31"}},
    #            {"data_path": SHIELD_DATA_PATH, "subset": {"start_time": "2011-01-01"}},
    #        ],
    #        "strict_ensemble": False,
    #    },
    # },
}


def write_config_dataset(config: Dict[str, Any]):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, DATASET_CONFIG_FILENAME)
        with open(filepath, "w") as f:
            yaml.safe_dump(config, f)
        dataset_name = "ace-inference-config-" + str(uuid.uuid4())[:8]
        dataset = client.dataset.create(dataset_name, filepath)
    return dataset


def get_experiment_spec(name: str, config: Dict[str, Any], image_name=IMAGE_NAME):
    """Given a dict representing the training configuration, return a beaker experiment spec."""
    config_dataset = write_config_dataset(config)
    env_vars = [
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key"),
        beaker.EnvVar(name="WANDB_JOB_TYPE", value="training"),
        beaker.EnvVar(name="WANDB_NAME", value=name),
    ]
    datasets = [
        beaker.DataMount(
            source=beaker.DataSource(beaker=config_dataset.id),
            mount_path=DATASET_CONFIG_MOUNTPATH,
        ),
        beaker.DataMount(
            source=beaker.DataSource(beaker=STATS_DATASET_NAME),
            mount_path="/statsdata",
        ),
        beaker.DataMount(
            mount_path="/climate-default",
            source=beaker.DataSource(weka="climate-default"),
        ),
    ]
    spec = beaker.ExperimentSpec(
        budget="ai2/climate",
        description="Do 10-year inference with ACE2 model trained on ERA5.",
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
        config = {**base_config, **overlay}
        print(f"Validating config for experiment {name}.")
        print(f"Config that is being validated:\n{config}")
        dacite.from_dict(fme.ace.TrainConfig, config, config=dacite.Config(strict=True))
    print("All configs are valid. Starting experiment submission.")
    for name, overlay in EXPERIMENT_OVERLAYS.items():
        config = {**base_config, **overlay}
        print(f"Creating experiment {name}.")
        spec = get_experiment_spec(name, config)
        try:
            experiment = client.experiment.create(name, spec, workspace="ai2/ace")
            msg = f"Experiment {name} created. https://beaker.org/ex/{experiment.id}"
            print(msg)
        except beaker.exceptions.ExperimentConflict:
            print(
                f"Failed to create experiment {name} because it already exists. "
                "Skipping experiment creation. If you want to submit this experiment, "
                "delete the existing experiment with the same name, or rename the new "
                "experiment."
            )
