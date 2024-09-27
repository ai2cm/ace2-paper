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
    "10yr-IC0": "tq6ljtf0",  # this is same as "10yr" run above, just used different data writer outputs
    "10yr-IC1": "m6qu89d2",
    "10yr-IC2": "h4oj4oc2",
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
CONSTRAINT_ABLATION_INFERENCE_WANDB_RUN_IDS = {
    "No constraints-IC0": "urr9bmrm",
    "No constraints-IC1": "nngxzfk3",
    "No constraints-IC2": "nr62yfwd",
    "Dry air-IC0": "ydtpd9x2",
    "Dry air-IC1": "09dhvite",
    "Dry air-IC2": "tk1qe8xq",
    "Dry air + moisture-IC0": "ejv18tb1",
    "Dry air + moisture-IC1": "m2na9qzb",
    "Dry air + moisture-IC2": "vndqpyhf",
}
