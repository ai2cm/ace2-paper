WANDB_ENTITY = "ai2cm"
WANDB_PROJECT = "ace"

# training job wandb IDs
ERA5_TRAINING_RUN_WANDB_IDS = {
    "rs0": "2g4hd4u5",
    "rs1": "o952oyir",
    "rs2": "593xn6l8",
    "rs3": "380pn4sr",
}
ERA5_BEST_RUN_WANDB_ID = ERA5_TRAINING_RUN_WANDB_IDS["rs2"]

# inference runs for physical constraints ablation
# TODO: redo this evaluator runs and regenerate the plots
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
