"""This script is used to get the wandb ids of the runs that we want to analyze."""

import beaker
import yaml
import utils

# the beaker experiment names are taken from these scripts
# ../experiments/evaluator-era5-main/run.py
# ../experiments/evaluator-era5-main/run-segmented.py
# ../experiments/evaluator-era5-main/run-rs-comparison.py
# ../experiments/evaluator-shield-amip-1deg/run.py

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
    "shield-amip-1deg-ace2-inference-10yr-IC0",
    "shield-amip-1deg-ace2-inference-10yr-IC1",
    "shield-amip-1deg-ace2-inference-10yr-IC2",
    "shield-amip-1deg-ace2-inference-81yr-IC0",
    "shield-amip-1deg-ace2-inference-81yr-IC1",
    "shield-amip-1deg-ace2-inference-81yr-IC2",
    "shield-amip-1deg-reference-inference-10yr",
    "shield-amip-1deg-reference-inference-81yr",
    "shield-amip-IC1-vs-era5-10yr",
    "shield-amip-IC2-vs-era5-10yr",
    "shield-amip-IC1-vs-era5-81yr",
    "shield-amip-IC2-vs-era5-81yr",
    "shield-amip-c96-vs-c24-4deg-10yr-IC0",
    "shield-amip-c96-vs-c24-4deg-10yr-IC1",
    "shield-amip-c96-vs-c24-4deg-81yr-IC0",
    "shield-amip-c96-vs-c24-4deg-81yr-IC1",
    "shield-amip-1deg-ace2-inference-10yr-RS0-IC0",
    "shield-amip-1deg-ace2-inference-10yr-RS0-IC1",
    "shield-amip-1deg-ace2-inference-10yr-RS0-IC2",
    "shield-amip-1deg-ace2-inference-81yr-RS0-IC0",
    "shield-amip-1deg-ace2-inference-81yr-RS0-IC1",
    "shield-amip-1deg-ace2-inference-81yr-RS0-IC2",
    "shield-amip-1deg-ace2-inference-10yr-RS1-IC0",
    "shield-amip-1deg-ace2-inference-10yr-RS1-IC1",
    "shield-amip-1deg-ace2-inference-10yr-RS1-IC2",
    "shield-amip-1deg-ace2-inference-81yr-RS1-IC0",
    "shield-amip-1deg-ace2-inference-81yr-RS1-IC1",
    "shield-amip-1deg-ace2-inference-81yr-RS1-IC2",
    "shield-amip-1deg-ace2-inference-10yr-RS2-IC0",
    "shield-amip-1deg-ace2-inference-10yr-RS2-IC1",
    "shield-amip-1deg-ace2-inference-10yr-RS2-IC2",
    "shield-amip-1deg-ace2-inference-81yr-RS2-IC0",
    "shield-amip-1deg-ace2-inference-81yr-RS2-IC1",
    "shield-amip-1deg-ace2-inference-81yr-RS2-IC2",
    "shield-amip-1deg-ace2-inference-10yr-RS3-IC0",
    "shield-amip-1deg-ace2-inference-10yr-RS3-IC1",
    "shield-amip-1deg-ace2-inference-10yr-RS3-IC2",
    "shield-amip-1deg-ace2-inference-81yr-RS3-IC0",
    "shield-amip-1deg-ace2-inference-81yr-RS3-IC1",
    "shield-amip-1deg-ace2-inference-81yr-RS3-IC2",
    "shield-amip-1deg-ace-climsst-inference-10yr-IC0",
    "shield-amip-1deg-ace-climsst-inference-10yr-IC1",
    "shield-amip-1deg-ace-climsst-inference-10yr-IC2",
    "shield-amip-1deg-ace-climsst-inference-81yr-IC0",
    "shield-amip-1deg-ace-climsst-inference-81yr-IC1",
    "shield-amip-1deg-ace-climsst-inference-81yr-IC2",
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