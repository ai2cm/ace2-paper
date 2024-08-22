WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "ace"

# training job wandb IDs
ERA5_TRAINING_RUN_WANDB_IDS = {
    "rs0": "2g4hd4u5",
    "rs1": "o952oyir",
    "rs2": "593xn6l8",
    "rs3": "380pn4s",
}
ERA5_BEST_RUN_WANDB_ID = ERA5_TRAINING_RUN_WANDB_IDS["rs2"]

# inference job wandb IDs using best checkpoint from above
ERA5_BEST_INFERENCE_WANDB_RUN_IDS = {
    "80yr": "q0ozc0o2",
    "10yr": "yl839jou",
    "15day": "cv2dfqs4",
    "100day": "6dood5ee",
}

# 'dataset comparison' run of ERA5 data against itself
ERA5_DATA_RUN_WANDB_ID = "y8njbnnb"

# inference job wandb IDs using best checkpoint
SHiELD_AMIP_1DEG_BEST_INFERENCE_10YR_WANDB_RUN_IDS = {
    "IC0": "5cz0zg44",
    "IC1": "s2mcmjro",
    "IC2": "mo9kma4p",
}
SHiELD_AMIP_1DEG_BEST_INFERENCE_82YR_WANDB_RUN_IDS = {
    "IC0": "bpbrb1nr",
    "IC1": "bcq9qmiw",
    "IC2": "gl7ng3hr", 
}

# 'dataset comparison' runs of SHiELD-AMIP IC0001 against IC0002 
SHiELD_AMIP_1DEG_REFERENCE_10YR_WANDB_RUN_ID = "x6fnlmfi"
SHiELD_AMIP_1DEG_REFERENCE_82YR_WANDB_RUN_ID = "bb1icrnr"

# 'dataset comparison' runs of SHiELD-AMIP ICs against ERA5
SHiELD_AMIP_ERA5_1DEG_COMPARISON_10YR_WANDB_RUN_IDS = {
    "IC0": "vium678t",
    "IC1": "yde9orc7",
}
SHiELD_AMIP_ERA5_1DEG_COMPARISON_82YR_WANDB_RUN_IDS = {
    "IC0": "hwhy4ce9",
    "IC1": "jm2or9pv",
}

CLIMSST_DEG_WANDB_RUN_IDS = {
    "IC0": "g87nkb9w",
    "IC1": "vbf7sq8n",
    "IC2": "s474gx2h"
}

# inference summary at 1deg
INFERENCE_COMPARISON_1DEG = {
    "10yr": {
        "ERA5-ACEv2": ERA5_BEST_INFERENCE_WANDB_RUN_IDS["10yr"],
        "SHiELD-AMIP-ACEv2": SHiELD_AMIP_1DEG_BEST_INFERENCE_10YR_WANDB_RUN_IDS,
        "SHiELD-AMIP-reference": SHiELD_AMIP_1DEG_REFERENCE_10YR_WANDB_RUN_ID,
        "SHiELD-AMIP-vs.-ERA5": SHiELD_AMIP_ERA5_1DEG_COMPARISON_10YR_WANDB_RUN_IDS,
    },
    "82yr": {
        "ERA5-ACEv2": ERA5_BEST_INFERENCE_WANDB_RUN_IDS["80yr"],
        "SHiELD-AMIP-ACEv2": SHiELD_AMIP_1DEG_BEST_INFERENCE_82YR_WANDB_RUN_IDS,
        "SHiELD-AMIP-reference": SHiELD_AMIP_1DEG_REFERENCE_82YR_WANDB_RUN_ID,
        "SHiELD-AMIP-vs.-ERA5": SHiELD_AMIP_ERA5_1DEG_COMPARISON_82YR_WANDB_RUN_IDS,
    }
}

# physical constants

SECONDS_PER_DAY = 86_400