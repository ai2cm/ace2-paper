import yaml

WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "ace"

WANDB_ID_FILE = "./wandb_ids.yaml"

# training job wandb IDs
ERA5_TRAINING_RUN_WANDB_IDS = {
    "rs0": "2g4hd4u5",
    "rs1": "o952oyir",
    "rs2": "593xn6l8",
    "rs3": "380pn4sr",
}
SHiELD_TRAINING_RUN_WANDB_IDS = {
    "rs0": "hfa38ef6",
    "rs1": "wjgsqy7x",
    "rs2": "qf8e8qy4",
    "rs3": "men5bnw6",
}

with open(WANDB_ID_FILE, "r") as f:
    wandb_ids = yaml.safe_load(f)


# this is brittle but works with current set of names
def get_runs_subset(all_runs, name_key):
    return {
        f"IC{k.split(name_key)[1][:1]}": v
        for k, v in all_runs.items()
        if k.startswith(name_key)
    }


# SHiELD evaluation wandb IDs using best checkpoint, RS2
ERA5_BEST_INFERENCE_10YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "era5-co2-10yr-RS2-IC"
)
ERA5_BEST_INFERENCE_81YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "era5-co2-81yr-RS2-IC"
)
ERA5_BEST_INFERENCE_1YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "era5-co2-1yr-2020-RS2-IC"
)

# SHiELD evaluation wandb IDs using best checkpoint, RS2
SHiELD_AMIP_1DEG_BEST_INFERENCE_10YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-10yr-IC"
)
SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-81yr-IC"
)

# SHiELD evaluation wandb IDs using other RS checkpoints
SHiELD_AMIP_1DEG_RS0_10YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-10yr-RS0-IC"
)
SHiELD_AMIP_1DEG_RS0_81YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-81yr-RS0-IC"
)
SHiELD_AMIP_1DEG_RS1_10YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-10yr-RS1-IC"
)
SHiELD_AMIP_1DEG_RS1_81YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-81yr-RS1-IC"
)
SHiELD_AMIP_1DEG_RS3_10YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-10yr-RS3-IC"
)
SHiELD_AMIP_1DEG_RS3_81YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-81yr-RS3-IC"
)

# 'dataset comparison' runs of SHiELD-AMIP IC0001 against IC0002
SHiELD_AMIP_1DEG_REFERENCE_10YR_WANDB_RUN_IDS = {
    "IC0": wandb_ids["shield-amip-1deg-reference-inference-10yr-1941"],
    "IC1": wandb_ids["shield-amip-1deg-reference-inference-10yr-1951"],
    "IC2": wandb_ids["shield-amip-1deg-reference-inference-10yr-1961"],
    "IC3": wandb_ids["shield-amip-1deg-reference-inference-10yr-1971"],
    "IC4": wandb_ids["shield-amip-1deg-reference-inference-10yr-1981"],
    "IC5": wandb_ids["shield-amip-1deg-reference-inference-10yr-1991"],
    "IC6": wandb_ids["shield-amip-1deg-reference-inference-10yr"],
    "IC7": wandb_ids["shield-amip-1deg-reference-inference-10yr-2011"],
}
SHiELD_AMIP_1DEG_REFERENCE_81YR_WANDB_RUN_ID = {
    "IC0": wandb_ids["shield-amip-1deg-reference-inference-81yr"],
}

# 'dataset comparison' runs of SHiELD-AMIP ICs against ERA5
SHiELD_AMIP_ERA5_1DEG_COMPARISON_10YR_WANDB_RUN_IDS = {
    "IC0": wandb_ids["shield-amip-IC1-vs-era5-10yr"],
    "IC1": wandb_ids["shield-amip-IC2-vs-era5-10yr"],
}
SHiELD_AMIP_ERA5_1DEG_COMPARISON_81YR_WANDB_RUN_IDS = {
    "IC0": wandb_ids["shield-amip-IC1-vs-era5-81yr"],
    "IC1": wandb_ids["shield-amip-IC2-vs-era5-81yr"],
}

# climSST ACE baseline
CLIMSST_DEG_10YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace-climsst-inference-10yr-IC"
)
CLIMSST_DEG_81YR_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace-climsst-inference-81yr-IC"
)

# ACE2-SHiELD CO2 sensitivity runs
SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_FIXEDCO2_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-81yr-fixedCO2-IC"
)
SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_NOCO2_WANDB_RUN_IDS = get_runs_subset(
    wandb_ids, "shield-amip-1deg-ace2-inference-81yr-noCO2-RS1-IC"
)

# inference summary at 1deg
INFERENCE_COMPARISON_1DEG = {
    "10yr": {
        "ACE2-ERA5": ERA5_BEST_INFERENCE_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD": SHiELD_AMIP_1DEG_BEST_INFERENCE_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS0": SHiELD_AMIP_1DEG_RS0_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS1": SHiELD_AMIP_1DEG_RS1_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS3": SHiELD_AMIP_1DEG_RS3_10YR_WANDB_RUN_IDS,
        "SHiELD-reference": SHiELD_AMIP_1DEG_REFERENCE_10YR_WANDB_RUN_IDS,
        "ACE-climSST": CLIMSST_DEG_10YR_WANDB_RUN_IDS,
        "SHiELD-vs.-ERA5": SHiELD_AMIP_ERA5_1DEG_COMPARISON_10YR_WANDB_RUN_IDS,
    },
    "81yr": {
        "ACE2-ERA5": ERA5_BEST_INFERENCE_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD": SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS0": SHiELD_AMIP_1DEG_RS0_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS1": SHiELD_AMIP_1DEG_RS1_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS3": SHiELD_AMIP_1DEG_RS3_81YR_WANDB_RUN_IDS,
        "SHiELD-reference": SHiELD_AMIP_1DEG_REFERENCE_81YR_WANDB_RUN_ID,
        "ACE-climSST": CLIMSST_DEG_81YR_WANDB_RUN_IDS,
        "SHiELD-vs.-ERA5": SHiELD_AMIP_ERA5_1DEG_COMPARISON_81YR_WANDB_RUN_IDS,
    },
}

# CO2 sensitivity
CO2_SENSITIVITY_ACE2_SHiELD_1DEG = {
    "historical-CO2": SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_WANDB_RUN_IDS,
    "fixed-CO2": SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_FIXEDCO2_WANDB_RUN_IDS,
    "no-CO2": SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_NOCO2_WANDB_RUN_IDS,
    "SHiELD-reference": SHiELD_AMIP_1DEG_REFERENCE_81YR_WANDB_RUN_ID,
}

ENSO_DEEP_DIVE = {
    "10yr": {
        "ACE2-ERA5": {
            "IC0": wandb_ids["era5-co2-10yr-RS2-IC0-monthly-output"],
            "IC1": wandb_ids["era5-co2-10yr-RS2-IC1-monthly-output"],
            "IC2": wandb_ids["era5-co2-10yr-RS2-IC2-monthly-output"],
        },
        "ACE2-SHiELD": {
            "IC0": wandb_ids["shield-amip-1deg-ace2-inference-10yr-IC0-monthly"],
            "IC1": wandb_ids["shield-amip-1deg-ace2-inference-10yr-IC1-monthly"],
            "IC2": wandb_ids["shield-amip-1deg-ace2-inference-10yr-IC2-monthly"],
        },
    },
    "81yr": {
        "ACE2-ERA5": {
            "IC0": wandb_ids["era5-co2-81yr-RS2-IC0-monthly-output"],
        },
        "ACE2-SHiELD": {
            "IC0": wandb_ids["shield-amip-1deg-ace2-inference-81yr-IC0-monthly"],
        },
    },
}

PERTURBED_SSTS = {
    "ACE2-SHiELD": {
        0.0: wandb_ids["shield-amip-1deg-ace2-inference-perturbed-30yr-ms-0p0-IC0"],
        0.5: wandb_ids["shield-amip-1deg-ace2-inference-perturbed-30yr-ms-0p5-IC0"],
        1.0: wandb_ids["shield-amip-1deg-ace2-inference-perturbed-30yr-ms-1p0-IC0"],
        2.0: wandb_ids["shield-amip-1deg-ace2-inference-perturbed-30yr-ms-2p0-IC0"],
        4.0: wandb_ids["shield-amip-1deg-ace2-inference-perturbed-30yr-ms-4p0-IC0"],
    },
    "ACE2-ERA5": {
        0.0: wandb_ids["era5-ace2-inference-perturbed-30yr-ms2-0p0-IC0"],
        0.5: wandb_ids["era5-ace2-inference-perturbed-30yr-ms2-0p5-IC0"],
        1.0: wandb_ids["era5-ace2-inference-perturbed-30yr-ms2-1p0-IC0"],
        2.0: wandb_ids["era5-ace2-inference-perturbed-30yr-ms2-2p0-IC0"],
        4.0: wandb_ids["era5-ace2-inference-perturbed-30yr-ms2-4p0-IC0"],
    }
}

# physical constants
SECONDS_PER_DAY = 86_400
