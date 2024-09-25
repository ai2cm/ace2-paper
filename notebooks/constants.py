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
    "81yr-IC0": "1zj3396n",
    "81yr-IC1": "0wet2ot4",
    "81yr-IC2": "g7tqggai",
    "10yr": "cmjkmtnw",
    "30yr": "w37x2qud",
    "15day": "me15actr",
    "100day": "oghlard3",
    "10yr-segmented": "5wsnbpl5",  # TODO: rerun with recent image
    "10yr-climSST": "c5e2dt6g",  # TODO: rerun with recent image
    "10yr-with-precip": "tq6ljtf0",
}

# 'dataset comparison' run of ERA5 data against itself
ERA5_DATA_RUN_WANDB_ID = "gxri8ksf"


# inference runs for physical constraints ablation
# TODO: rerun these experiments with latest image
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
