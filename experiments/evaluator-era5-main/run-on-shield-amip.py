# requires beaker-py, install with
# pip install -U beaker-py

import os
import tempfile
import uuid
from typing import Any, Dict

import beaker
import dacite
import fme
import yaml

IMAGE_NAME = "brianhenn/fme-944f247d"
TRAINED_MODEL_DATASET_ID = "01J4MT10JPQ8MFA41F2AXGFYJ9"
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"
SHIELD_AMIP_DATA_PATH = "/climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0002"
ERA5_DATA_PATH = "/climate-default/2024-06-20-era5-1deg-8layer-1940-2022-netcdfs"

EXPERIMENT_OVERLAYS = {
    "ERA5-on-SHiELD-AMIP-81yr-RS2-IC0": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "dataset": {"data_path": SHIELD_AMIP_DATA_PATH},
        },
        "aggregator": {},
    },
    "ERA5-on-SHiELD-AMIP-81yr-RS2-IC1": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-02T12:00:00"]},
            "dataset": {"data_path": SHIELD_AMIP_DATA_PATH},
        },
        "aggregator": {},
    },
    "ERA5-on-SHiELD-AMIP-81yr-RS2-IC2": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-03T12:00:00"]},
            "dataset": {"data_path": SHIELD_AMIP_DATA_PATH},
        },
        "aggregator": {},
    },
    "ERA5-on-ERA5-81yr-RS2-IC0": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "dataset": {"data_path": ERA5_DATA_PATH},
        },
        "aggregator": {},
    },
    "ERA5-on-ERA5-81yr-RS2-IC1": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-02T12:00:00"]},
            "dataset": {"data_path": ERA5_DATA_PATH},
        },
        "aggregator": {},
    },
    "ERA5-on-ERA5-81yr-RS2-IC2": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-03T12:00:00"]},
            "dataset": {"data_path": ERA5_DATA_PATH},
        },
        "aggregator": {},
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
                    "fme.ace.inference",
                    f"{DATASET_CONFIG_MOUNTPATH}/{DATASET_CONFIG_FILENAME}",
                ],
                result=beaker.ResultSpec(path="/output"),
                resources=beaker.TaskResources(gpu_count=1, shared_memory="50GiB"),
                context=beaker.TaskContext(priority="high", preemptible=False),
                constraints=beaker.Constraints(
                    cluster=[
                        "ai2/jupiter-cirrascale-2",
                        "ai2/saturn-cirrascale",
                        "ai2/ceres-cirrascale",
                    ]
                ),
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
