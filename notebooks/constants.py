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

# corresponding beaker dataset IDs
#api = wandb.Api()
#beaker_client = beaker.Beaker.from_env()
#ERA5_BEST_INFERENCE_BEAKER_DATASET_IDS = {}
#for name, id in ERA5_BEST_INFERENCE_WANDB_RUN_IDS.items():
    # for now these runs are in different project, will change in future
#    run = api.run(f"{WANDB_ENTITY}/ace2/{id}")
#    beaker_experiment_id = run.config["environment"]["BEAKER_EXPERIMENT_ID"]
#    result_dataset = beaker_client.experiment.results(beaker_experiment_id)
#    ERA5_BEST_INFERENCE_BEAKER_DATASET_IDS[name] = result_dataset.id

#print(ERA5_BEST_INFERENCE_BEAKER_DATASET_IDS)