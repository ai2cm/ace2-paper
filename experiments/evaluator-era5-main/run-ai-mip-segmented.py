# do 2-year segmented into month-long segments to enable saving all the outputs
# necessary for the AI-MIP initial evaluation

import os
import tempfile
import uuid
from typing import Any, Dict

import beaker
import dacite
import fme
import yaml

IMAGE_NAME = "oliverwm/fme-926fd6e7"
TRAINED_MODEL_DATASET_ID = "01J4MT10JPQ8MFA41F2AXGFYJ9"
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "ai-mip-segmented-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"


# experiments defined by overlays which will overwrite the keys of the base config
EXPERIMENT_OVERLAYS = {
    "era5-co2-2yr-RS2-ai-mip-output-segmented-ni": {},
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
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key-ai2cm-sa"),
        beaker.EnvVar(name="WANDB_USERNAME", value="bhenn1983"),
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
                    "--segments",
                    "20",
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
            fme.ace.InferenceConfig, config, config=dacite.Config(strict=True)
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
