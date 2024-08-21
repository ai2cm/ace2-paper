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

IMAGE_NAME = "oliverwm/fme-7fc6b9f8"
TRAINED_MODEL_DATASET_ID = "01J5NADA3E9PPAAT63X010B3HB"
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"


# experiments defined by overlays which will overwrite the keys of the base config
DATA_PATH = "/climate-default/2024-06-20-era5-1deg-8layer-1940-2022-netcdfs"
EXPERIMENT_OVERLAYS = {
    "era5-co2-10yr-RS3": {
        "n_forward_steps": 14600,
        "loader": {
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
        "data_writer": {
            "save_prediction_files": True,
            "names": ["PRATEsfc"],
        },
    },
    "era5-co2-10yr-daily-output-RS3": {
        "n_forward_steps": 14600,
        "loader": {
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
        "data_writer": {
            "save_prediction_files": True,
            "names": ["eastward_wind_0"],
            "time_coarsen": {"coarsen_factor": 4},
        },
    },
    "era5-co2-30yr-daily-output-RS3": {
        "n_forward_steps": 43800,
        "loader": {
            "start_indices": {"times": ["1991-01-01T00:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
        "data_writer": {
            "save_prediction_files": True,
            "names": ["eastward_wind_0"],
            "time_coarsen": {"coarsen_factor": 4},
        },
    },
    "era5-co2-81yr-RS3-IC0": {
        "n_forward_steps": 118341,
        "aggregator": {"log_zonal_mean_images": False},
    },
    "era5-co2-81yr-RS3-IC1": {
        "n_forward_steps": 118341,
        "aggregator": {"log_zonal_mean_images": False},
        "loader": {
            "start_indices": {"times": ["1940-01-02T12:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
    },
    "era5-co2-81yr-RS3-IC2": {
        "n_forward_steps": 118341,
        "aggregator": {"log_zonal_mean_images": False},
        "loader": {
            "start_indices": {"times": ["1940-01-03T12:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
        "data_writer": {
            "save_prediction_files": False,
            "save_monthly_files": True,
            "names": ["total_water_path", "air_temperature_7", "eastward_wind_0"],
        },
    },
    "era5-truth-81yr": {
        "n_forward_steps": 118341,
        "aggregator": {"log_zonal_mean_images": False},
        "prediction_loader": {
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
    },
    "era5-co2-15day-2020-RC3": {
        "n_forward_steps": 60,
        "forward_steps_in_memory": 1,
        "loader": {
            "start_indices": {
                "first": 116881,
                "interval": 26,
                "n_initial_conditions": 50,
            },
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
    },
    "era5-co2-100day-2020-video-RC3": {
        "n_forward_steps": 400,
        "forward_steps_in_memory": 40,
        "aggregator": {"log_video": True, "log_histograms": True},
        "loader": {
            "start_indices": {"times": ["2020-08-20T00:00:00"]},
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
    },
}


def write_config_dataset(config: Dict[str, Any]):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, DATASET_CONFIG_FILENAME)
        with open(filepath, "w") as f:
            yaml.safe_dump(config, f)
        dataset_name = "ace-inference-config-" + str(uuid.uuid4())[:8]
        dataset = client.dataset.create(dataset_name, filepath)
    return dataset


def get_experiment_spec(
    name: str,
    config: Dict[str, Any],
    image_name=IMAGE_NAME,
    trained_model_dataset_id=TRAINED_MODEL_DATASET_ID,
):
    """Given a dict representing the inference configuration, return a beaker experiment spec."""
    config_dataset = write_config_dataset(config)
    env_vars = [
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key"),
        beaker.EnvVar(name="WANDB_JOB_TYPE", value="inference"),
        beaker.EnvVar(name="WANDB_NAME", value=name),
    ]
    datasets = [
        beaker.DataMount(
            source=beaker.DataSource(beaker=config_dataset.id),
            mount_path=DATASET_CONFIG_MOUNTPATH,
        ),
        beaker.DataMount(
            mount_path="/ckpt.tar",
            source=beaker.DataSource(beaker=trained_model_dataset_id),
            sub_path=f"training_checkpoints/{CHECKPOINT_NAME}",
        ),
        beaker.DataMount(
            mount_path="/climate-default",
            source=beaker.DataSource(weka="climate-default"),
        ),
    ]
    spec = beaker.ExperimentSpec(
        budget="ai2/climate",
        description="Do inference with ACE2 model trained on ERA5.",
        tasks=[
            beaker.TaskSpec(
                name=name,
                image=beaker.ImageSource(beaker=image_name),
                command=[
                    "python",
                    "-m",
                    "fme.ace.evaluator",
                    f"{DATASET_CONFIG_MOUNTPATH}/{DATASET_CONFIG_FILENAME}",
                ],
                result=beaker.ResultSpec(path="/output"),
                resources=beaker.TaskResources(gpu_count=1, shared_memory="50GiB"),
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
        dacite.from_dict(
            fme.ace.InferenceEvaluatorConfig, config, config=dacite.Config(strict=True)
        )
    print("All configs are valid. Starting experiment submission.")
    for name, overlay in EXPERIMENT_OVERLAYS.items():
        config = {**base_config, **overlay}
        print(f"Creating experiment {name}.")
        spec = get_experiment_spec(name, config)
        try:
            experiment = client.experiment.create(name, spec, workspace="ai2/ace")
            print(
                f"Experiment {name} created. See https://beaker.org/ex/{experiment.id}"
            )
        except beaker.exceptions.ExperimentConflict:
            print(
                f"Failed to create experiment {name} because it already exists. "
                "Skipping experiment creation. If you want to submit this experiment, "
                "delete the existing experiment with the same name, or rename the new "
                "experiment."
            )
