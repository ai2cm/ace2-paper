WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "ace"

# training job wandb IDs
ERA5_TRAINING_RUN_WANDB_IDS = {
    "rs0": "2g4hd4u5",
    "rs1": "o952oyir",
    "rs2": "593xn6l8",
    "rs3": "380pn4s",
}
ERA5_BEST_RUN_WANDB_ID = ERA5_TRAINING_RUN_WANDB_IDS["rs3"]

# inference job wandb IDs using best checkpoint from above
ERA5_BEST_INFERENCE_WANDB_RUN_IDS = {
    "81yr-IC0": "ffyjif5r",
    "81yr-IC1": "rsv9yjp1",
    "81yr-IC2": "k8hnc4sz",
    "10yr": "k5kh8cm7",
    "30yr": "ennqyul0",
    "15day": "6x44o2h0",
    "100day": "3ey9f2cg",
    "10yr-segmented": "5wsnbpl5",
}

# 'dataset comparison' run of ERA5 data against itself
ERA5_DATA_RUN_WANDB_ID = "y8njbnnb"
