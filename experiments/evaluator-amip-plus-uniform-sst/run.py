# requires beaker-py, install with
# pip install -U beaker-py

import beaker
import uuid
from typing import Dict, Any
import tempfile
import yaml
import os

IMAGE_NAME = "brianhenn/fme-926fd6e7"
ACE2_SHIELD_MODEL_DATASET_ID = "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt"
SHIELD_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0002"
ERA5_DATASET_PATH = "/climate-default/2024-06-20-era5-1deg-8layer-1940-2022-netcdfs"
IC_FILENAME = "1979010100.nc"
ACE2_ERA5_MODEL_DATASET_ID = "01J4MT10JPQ8MFA41F2AXGFYJ9"
REFERENCE_DATASET_PATH = "" # TBD
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"

PERTURBATIONS = {
    "0p0": 0.0,
    "0p5": 0.5,
    "1p0": 1.0,
    "2p0": 2.0,
    "4p0": 4.0,
}
INITIAL_CONDITIONS = {
    "IC0": "1979-01-01T00:00:00",
    "IC1": "1979-01-02T00:00:00",
    "IC2": "1979-01-03T00:00:00",
}

GROUP_TEMPLATE = "{model}-ace2-inference-perturbed-30yr-ms2-{group_suffix}"
NAME_TEMPLATE = "{group_name}-{experiment_suffix}"

HUNDRED_DAY_RUN_GROUP = "shield-amip-1deg-ace2-inference-perturbed-30yr-100d"
HUNDRED_DAY_RUN_NAME = f"{HUNDRED_DAY_RUN_GROUP}-4p0-IC0"
HUNDRED_DAY_OVERLAY = {
    "data_writer": {
        "save_monthly_files": False,
        "save_prediction_files": True,
    },
    "n_forward_steps": 400,
    "forcing_loader": {
        "perturbations": {
            "sst": [
                {
                    "name": "constant",
                    "config": {
                        "amplitude": 4.0,
                    },
                },
            ]
        },
    },
    "initial_condition": {
        "start_indices": {
            "times":
                [
                    "1979-01-01T00:00:00",
                ]
        }
    },
}

def get_experiment_overlay(
    perturbation: float,
    ic_date: str,
    dataset_dir: str,
    ic_filename: str=IC_FILENAME,
) -> Dict[str, Any]:
    return {
        "forcing_loader": {
            "dataset": {
                "data_path": dataset_dir
            },
            "perturbations": {
                "sst": [
                    {
                        "name": "constant",
                        "config": {
                            "amplitude": perturbation,
                        },
                    },
                ]
            },
        },
        "initial_condition": {
            "path": f"{dataset_dir}/{ic_filename}",
            "start_indices": {
                "times":
                    [
                        ic_date,
                    ]
            }
        }
    }


def merge_configs(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested configurations."""
    base_copy = base.copy() # don't modify the original base
    for k, v in new.items():
        if isinstance(v, dict):
            base_copy[k] = merge_configs(base_copy.get(k, {}), v)
        else:
            base_copy[k] = v
    return base_copy


def write_config_dataset(config: Dict[str, Any]) -> beaker.Dataset:
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, DATASET_CONFIG_FILENAME)
        with open(filepath, "w") as f:
            yaml.safe_dump(config, f)
        dataset_name = "ace-inference-config-" + str(uuid.uuid4())[:8]
        dataset = client.dataset.create(dataset_name, filepath)
    return dataset


def get_experiment_spec(
    group_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    trained_model_dataset_id: str,
    image_name: str=IMAGE_NAME,
) -> beaker.ExperimentSpec:
    """Given a dict representing the inference configuration, return a beaker experiment spec."""
    config_dataset = write_config_dataset(config)
    env_vars = [
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key-ai2cm-sa"),
        beaker.EnvVar(name="WANDB_JOB_TYPE", value="inference"),
        beaker.EnvVar(name="WANDB_NAME", value=experiment_name),
        beaker.EnvVar(name="WANDB_RUN_GROUP", value=group_name),
        beaker.EnvVar(name="WANDB_USERNAME", value="bhenn1983")
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
                name=experiment_name,
                image=beaker.ImageSource(beaker=image_name),
                command=[
                    "python",
                    "-m",
                    "fme.ace.inference",
                    f"{DATASET_CONFIG_MOUNTPATH}/{DATASET_CONFIG_FILENAME}",
                ],
                result=beaker.ResultSpec(path="/output"),
                resources=beaker.TaskResources(gpu_count=1, shared_memory="50GiB"),
                context=beaker.TaskContext(priority="high", preemptible=True),
                constraints=beaker.Constraints(cluster=["ai2/saturn-cirrascale"]),
                env_vars=env_vars,
                datasets=datasets,
            )
        ],
    )
    return spec

def try_submit_experiment(experiment_name: str, spec: beaker.ExperimentSpec):
    try:
        experiment = client.experiment.create(experiment_name, spec)
        print(
            f"Experiment {experiment_name} created. See https://beaker.org/ex/{experiment.id}"
        )
    except beaker.exceptions.ExperimentConflict:
        print(
            f"Failed to create experiment {experiment_name} because it already exists. "
            "Skipping experiment creation. If you want to submit this experiment, "
            "delete the existing experiment with the same name, or rename the new "
            "experiment."
        )



if __name__ == "__main__":
    client = beaker.Beaker.from_env()

    with open(LOCAL_BASE_CONFIG_FILENAME, "r") as f:
        base_config = yaml.safe_load(f)

    print(f"Creating experiment {HUNDRED_DAY_RUN_NAME}.")
    hundred_day_config = merge_configs(base_config, HUNDRED_DAY_OVERLAY)
    print(f"Config that is being submitted:\n{hundred_day_config}")
    hundred_day_spec = get_experiment_spec(HUNDRED_DAY_RUN_GROUP, HUNDRED_DAY_RUN_NAME, hundred_day_config, ACE2_SHIELD_MODEL_DATASET_ID)
    try_submit_experiment(HUNDRED_DAY_RUN_NAME, hundred_day_spec)

    for perturbation_name, perturbation in PERTURBATIONS.items():
        for model_name, model_id, dataset_dir in zip(
            ("shield-amip-1deg", "era5"),
            (ACE2_SHIELD_MODEL_DATASET_ID, ACE2_ERA5_MODEL_DATASET_ID),
            (SHIELD_DATASET_PATH, ERA5_DATASET_PATH),
        ): 
            perturbation_group_name = GROUP_TEMPLATE.format(model=model_name, group_suffix=perturbation_name)
            for ic_name, ic_date in INITIAL_CONDITIONS.items():
                ic_experiment_name = NAME_TEMPLATE.format(group_name=perturbation_group_name, experiment_suffix=ic_name)
                experiment_overlay = get_experiment_overlay(perturbation, ic_date, dataset_dir)
                config = merge_configs(base_config, experiment_overlay)
                print(f"Creating experiment {ic_experiment_name}.")
                print(f"Config that is being submitted:\n{config}")
                spec = get_experiment_spec(perturbation_group_name, ic_experiment_name, config, model_id)
                try_submit_experiment(ic_experiment_name, spec)