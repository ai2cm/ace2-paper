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

IMAGE_NAME = "oliverwm/fme-926fd6e7"
WORKSPACE = "ai2/ace"
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"

TRAINED_MODEL_DATASET_IDS = {
    "no-constraints": "01J61CG7N6XD6YWH2WSTP84JYG",  # https://wandb.ai/ai2cm/ace/runs/ohxkr4ya
    "dry-air": "01J5658PYCEDP60678ERMRPCVJ",  # https://wandb.ai/ai2cm/ace/runs/lu30xajn
    "dry-air-and-moisture": "01J52JFYZ78DAH1DTGW3YEVRYQ",  # https://wandb.ai/ai2cm/ace/runs/qf8e8qy4
}

# experiments defined by overlays which will overwrite the keys of the base config
EXPERIMENT_OVERLAYS = {
    "shield-amip-{constraint}-10yr-IC0-rerun": {"n_forward_steps": 14600},
    "shield-amip-{constraint}-10yr-IC1-rerun": {
        "n_forward_steps": 14600,
        "loader": {
            "start_indices": {"times": ["2001-01-02T00:00:00"]},
        },
    },
    "shield-amip-{constraint}-10yr-IC2-rerun": {
        "n_forward_steps": 14600,
        "loader": {
            "start_indices": {"times": ["2001-01-03T00:00:00"]},
        },
    },
}


def merge_configs(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested configurations."""
    base_copy = base.copy()  # don't modify the original base
    for k, v in new.items():
        if isinstance(v, dict):
            base_copy[k] = merge_configs(base_copy.get(k, {}), v)
        else:
            base_copy[k] = v
    return base_copy


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
    trained_model_dataset_id: str,
    image_name=IMAGE_NAME,
):
    """Given a dict representing the inference configuration, return a beaker experiment spec."""
    config_dataset = write_config_dataset(config)
    env_vars = [
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key-ai2cm-sa"),
        beaker.EnvVar(name="WANDB_JOB_TYPE", value="inference"),
        beaker.EnvVar(name="WANDB_NAME", value=name),
        beaker.EnvVar(name="WANDB_RUN_GROUP", value="shield-amip-ace2-inference"),
        beaker.EnvVar(name="WANDB_USERNAME", value="oliverwm"),
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
        description="Do inference with ACE2 model trained on SHiELD-AMIP.",
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
        config = merge_configs(base_config, overlay)
        print(f"Validating config for experiment {name}.")
        print(f"Config that is being validated:\n{config}")
        dacite.from_dict(
            fme.ace.InferenceEvaluatorConfig, config, config=dacite.Config(strict=True)
        )
    print("All configs are valid. Starting experiment submission.")
    for constraint, model_id in TRAINED_MODEL_DATASET_IDS.items():
        for name_template, overlay in EXPERIMENT_OVERLAYS.items():
            name = name_template.format(constraint=constraint)
            config = merge_configs(base_config, overlay)
            print(f"Creating experiment {name}.")
            spec = get_experiment_spec(name, config, trained_model_dataset_id=model_id)
            try:
                experiment = client.experiment.create(name, spec, workspace=WORKSPACE)
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
