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
    "10yr-climSST": "c5e2dt6g",
    "10yr-with-precip": "78bu4fkz",
}

# 'dataset comparison' run of ERA5 data against itself
ERA5_DATA_RUN_WANDB_ID = "y8njbnnb"


# inference runs for physical constraints ablation
CONSTRAINT_ABLATION_INFERENCE_WANDB_RUN_IDS = {
    "No constraints-IC0": "dtq1ckuv",
    "No constraints-IC1": "6di90zwa",
    "No constraints-IC2": "kuv55mog",
    "Dry air-IC0": "w8z4goal",
    "Dry air-IC1": "yrbxabpf",
    "Dry air-IC2": "30ezr5an",
    "Dry air + moisture-IC0": "p4r9r3pi",
    "Dry air + moisture-IC1": "1xzj4pg1",
    "Dry air + moisture-IC2": "5lrii540",
}
