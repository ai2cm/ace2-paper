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

# SHiELD evaluation wandb IDs using best checkpoint, RS3
SHiELD_AMIP_1DEG_BEST_INFERENCE_10YR_WANDB_RUN_IDS = {
    "IC0": "a88aanz5",
    "IC1": "nj0ucjh6",
    "IC2": "de4mczvq",
}
SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_WANDB_RUN_IDS = {
    "IC0": "awvfqq9f",
    "IC1": "onovz8yp",
    "IC2": "qnsm54zm", 
}

# SHiELD evaluation wandb IDs using other RS checkpoints
SHiELD_AMIP_1DEG_RS0_10YR_WANDB_RUN_IDS = {
    "IC0": "cwwcu5kv",
    "IC1": "bermtwic",
    "IC2": "s8e6xsbk",
}
SHiELD_AMIP_1DEG_RS0_81YR_WANDB_RUN_IDS = {
    "IC0": "f266tolj",
    "IC1": "5mq7btcy",
    "IC2": "snkmw3d9", 
}
SHiELD_AMIP_1DEG_RS1_10YR_WANDB_RUN_IDS = {
    "IC0": "fig0aqhc",
    "IC1": "449ydn8p",
    "IC2": "dsv1xhpl",
}
SHiELD_AMIP_1DEG_RS1_81YR_WANDB_RUN_IDS = {
    "IC0": "qlm1etey",
    "IC1": "3bfxv1i9",
    "IC2": "n3u43uoa", 
}
SHiELD_AMIP_1DEG_RS2_10YR_WANDB_RUN_IDS = {
    "IC0": "mfdzt2o4",
    "IC1": "1ufjnbqp",
    "IC2": "rtnixrkw",
}
SHiELD_AMIP_1DEG_RS2_81YR_WANDB_RUN_IDS = {
    "IC0": "q41rz6p2",
    "IC1": "wreqe5gq",
    "IC2": "r2cpu2bp", 
}


# 'dataset comparison' runs of SHiELD-AMIP IC0001 against IC0002 
SHiELD_AMIP_1DEG_REFERENCE_10YR_WANDB_RUN_ID = {
    "IC0": "muswh21i"
}
SHiELD_AMIP_1DEG_REFERENCE_81YR_WANDB_RUN_ID = {
    "IC0": "ce2hgxs8"
}

# 'dataset comparison' runs of SHiELD-AMIP ICs against ERA5
SHiELD_AMIP_ERA5_1DEG_COMPARISON_10YR_WANDB_RUN_IDS = {
    "IC0": "2nw8gi5i",
    "IC1": "j83sbri7",
}
SHiELD_AMIP_ERA5_1DEG_COMPARISON_81YR_WANDB_RUN_IDS = {
    "IC0": "klnr8wwa",
    "IC1": "8t90aafd",
}

# climSST ACE baseline
CLIMSST_DEG_10YR_WANDB_RUN_IDS = {
    "IC0": "b76h3n7y",
    "IC1": "jlhmg5fp",
    "IC2": "9wyaoyt9"
}
CLIMSST_DEG_81YR_WANDB_RUN_IDS = {
    "IC0": "lg7a92g5",
    "IC1": "oydw8d94",
    "IC2": "saj7n8gd"
}

# inference summary at 1deg
INFERENCE_COMPARISON_1DEG = {
    "10yr": {
        "ACE2-ERA5": {
            k: v
            for k, v in ERA5_BEST_INFERENCE_WANDB_RUN_IDS.items()
            if k.startswith('10yr-IC')
        },
        "ACE2-SHiELD": SHiELD_AMIP_1DEG_BEST_INFERENCE_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS0": SHiELD_AMIP_1DEG_RS0_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS1": SHiELD_AMIP_1DEG_RS1_10YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS2": SHiELD_AMIP_1DEG_RS2_10YR_WANDB_RUN_IDS,
        "SHiELD-reference": SHiELD_AMIP_1DEG_REFERENCE_10YR_WANDB_RUN_ID,
        "ACE-climSST": CLIMSST_DEG_10YR_WANDB_RUN_IDS,
        "SHiELD-vs.-ERA5": SHiELD_AMIP_ERA5_1DEG_COMPARISON_10YR_WANDB_RUN_IDS,
    },
    "81yr": {
        "ACE2-ERA5":  {
            k: v
            for k, v in ERA5_BEST_INFERENCE_WANDB_RUN_IDS.items()
            if k.startswith('81yr-IC')
        },
        "ACE2-SHiELD": SHiELD_AMIP_1DEG_BEST_INFERENCE_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS0": SHiELD_AMIP_1DEG_RS0_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS1": SHiELD_AMIP_1DEG_RS1_81YR_WANDB_RUN_IDS,
        "ACE2-SHiELD-RS2": SHiELD_AMIP_1DEG_RS2_81YR_WANDB_RUN_IDS,
        "SHiELD-reference": SHiELD_AMIP_1DEG_REFERENCE_81YR_WANDB_RUN_ID,
        "ACE-climSST": CLIMSST_DEG_81YR_WANDB_RUN_IDS,
        "SHiELD-vs.-ERA5": SHiELD_AMIP_ERA5_1DEG_COMPARISON_81YR_WANDB_RUN_IDS,
    }
}

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


# physical constants
SECONDS_PER_DAY = 86_400
