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

IMAGE_NAME = "brianhenn/fme-f3337723"
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"


# experiments defined by overlays which will overwrite the keys of the base config
DATA_PATH = "/climate-default/2024-06-20-era5-1deg-8layer-1940-2022-netcdfs"
TRAINED_MODEL_DATASET_IDS = {
    "rs0": "01J4831KMJ5DZEE5RX4HGC1DB1",
    "rs1": "01J4874GRXA4XZAPNJY3K9NQNR",
    "rs2": "01J4MT10JPQ8MFA41F2AXGFYJ9",
    "rs3": "01J5NADA3E9PPAAT63X010B3HB",
}
EXPERIMENT_OVERLAYS = {
    "era5-co2-60yr-{rs}-ni": {
        "n_forward_steps": 7300,
        "forward_steps_in_memory": 5,
        "aggregator": {
            "log_histograms": False,
            "log_global_mean_time_series": False,
            "log_global_mean_norm_time_series": False,
            "log_zonal_mean_images": False,
        },
        "loader": {
            "start_indices": {
                "times": [
                    "1940-01-01T12:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]
            },
            "dataset": {"data_path": DATA_PATH},
            "num_data_workers": 8,
        },
    }
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
    trained_model_dataset_id,
    image_name=IMAGE_NAME,
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
    for name_template, overlay in EXPERIMENT_OVERLAYS.items():
        for rs, trained_model_dataset_id in TRAINED_MODEL_DATASET_IDS.items():
            name = name_template.format(rs=rs)
            config = {**base_config, **overlay}
            print(f"Creating experiment {name}.")
            spec = get_experiment_spec(name, config, trained_model_dataset_id)
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
