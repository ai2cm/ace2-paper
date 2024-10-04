"""This script is used to get the wandb ids of the runs that we want to analyze."""

import beaker
import yaml
import utils

# the beaker experiment names are taken from these scripts
# ../experiments/evaluator-era5-main/run.py
# ../experiments/evaluator-era5-main/run-segmented.py
# ../experiments/evaluator-era5-main/run-rs-comparison.py

beaker_experiment_names = [
    "era5-co2-10yr-RS2-IC0-ni",
    "era5-co2-10yr-RS2-IC1-ni",
    "era5-co2-10yr-RS2-IC2-ni",
    "era5-co2-10yr-daily-output-RS2-ni",
    "era5-co2-30yr-daily-output-RS2-ni",
    "era5-co2-81yr-RS2-IC0-ni",
    "era5-co2-81yr-RS2-IC1-ni",
    "era5-co2-81yr-RS2-IC2-ni",
    "era5-truth-81yr-ni",
    "era5-truth-10yr-ni",
    "era5-co2-15day-2020-RS2-ni",
    "era5-co2-100day-2020-video-RS2-wd-ni",
    "era5-co2-10yr-RS2-segmented-ni",
    "era5-co2-10yr-RS2-clim-forcing-ni",
    "era5-co2-60yr-rs0-ni",
    "era5-co2-60yr-rs1-ni",
    "era5-co2-60yr-rs2-ni",
    "era5-co2-60yr-rs3-ni",
]

# assuming all runs are using same wandb entity and project
wandb_ids = {}
for name in beaker_experiment_names:
    entity, project, id_ = utils.beaker_experiment_to_wandb(name)
    if entity is not None:
        assert entity == "ai2cm"
        assert project == "ace"
    wandb_ids[name] = id_

with open("wandb_ids.yaml", "w") as f:
    yaml.safe_dump(wandb_ids, f)