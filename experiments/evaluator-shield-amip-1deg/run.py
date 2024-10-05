# requires beaker-py, install with
# pip install -U beaker-py

import beaker
import uuid
from typing import Dict, Any
import tempfile
import yaml
import os
import fme
import dacite

IMAGE_NAME = "brianhenn/fme-926fd6e7"
TRAINED_MODEL_DATASET_ID = "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt"
REFERENCE_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0001"
TARGET_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c96-1deg-shield-amip-ensemble-dataset/netCDFs/ic_0002"
C24_4DEG_IC0_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c24-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0001"
C24_4DEG_IC1_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c24-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0002"
C96_4DEG_IC0_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0001"
C96_4DEG_IC1_DATASET_PATH = "/climate-default/2024-07-24-vertically-resolved-c96-4deg-shield-amip-ensemble-dataset/netCDFs/ic_0002"
ERA5_DATASET_PATH = "/climate-default/2024-06-20-era5-1deg-8layer-1940-2022-netcdfs"
CHECKPOINT_NAME = "best_inference_ckpt.tar"
LOCAL_BASE_CONFIG_FILENAME = "base-config.yaml"
DATASET_CONFIG_FILENAME = "config.yaml"
DATASET_CONFIG_MOUNTPATH = "/configmount"


# experiments defined by overlays which will overwrite the keys of the base config
EXPERIMENT_OVERLAYS = {
    "shield-amip-1deg-ace2-inference-10yr-IC0": {
        "n_forward_steps": 14600,
    },
    "shield-amip-1deg-ace2-inference-10yr-IC1": {
        "n_forward_steps": 14600,
        "loader": {
            "start_indices": {"times": ["2001-01-02T00:00:00"]},
        },
    },
    "shield-amip-1deg-ace2-inference-10yr-IC2": {
        "n_forward_steps": 14600,
        "loader": {
            "start_indices": {"times": ["2001-01-03T00:00:00"]},
        },
    },
    "shield-amip-1deg-ace2-inference-81yr-IC0": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "global_mean_co2",
            ],
        }
    },
    "shield-amip-1deg-ace2-inference-81yr-IC1": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-02T12:00:00"]},
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "global_mean_co2",
            ],
        }
    },
    "shield-amip-1deg-ace2-inference-81yr-IC2": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-03T12:00:00"]},
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "global_mean_co2",
            ],
        }
    },
    "shield-amip-1deg-ace2-inference-81yr-fixedCO2-IC0": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "dataset": {
                "overwrite": {
                    "constant": {
                        "global_mean_co2": 0.00030729,
                    },
                },
            },
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "global_mean_co2",
            ],
        }
    },
    "shield-amip-1deg-ace2-inference-81yr-fixedCO2-IC1": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-02T12:00:00"]},
            "dataset": {
                "overwrite": {
                    "constant": {
                        "global_mean_co2": 0.00030729,
                    },
                },
            },
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "global_mean_co2",
            ],
        }
    },
    "shield-amip-1deg-ace2-inference-81yr-fixedCO2-IC2": {
        "n_forward_steps": 118341,
        "loader": {
            "start_indices": {"times": ["1940-01-03T12:00:00"]},
            "dataset": {
                "overwrite": {
                    "constant": {
                        "global_mean_co2": 0.00030729,
                    },
                },
            },
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "global_mean_co2",
            ],
        }
    },
    "shield-amip-1deg-reference-inference-10yr": {
        "n_forward_steps": 14600,
        "prediction_loader": {
            "dataset": {"data_path": REFERENCE_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "num_data_workers": 8,
        }
    },
    "shield-amip-1deg-reference-inference-81yr": {
        "n_forward_steps": 118341,
        "loader": {"start_indices": {"times": ["1940-01-01T12:00:00"]}},
        "prediction_loader": {
            "dataset": {"data_path": REFERENCE_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "num_data_workers": 8,
        },
        "data_writer": {
            "save_monthly_files": True,
            "names": [
                "TMP2m",
                "SHTFLsfc",
                "surface_temperature",
                "ocean_fraction",
            ],
        },
    },
    'shield-amip-IC1-vs-era5-10yr': {
        "n_forward_steps": 14600,
        "loader": {
            "dataset": {"data_path": ERA5_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": REFERENCE_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-IC2-vs-era5-10yr': {
        "n_forward_steps": 14600,
        "loader": {
            "dataset": {"data_path": ERA5_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": TARGET_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-IC1-vs-era5-81yr': {
        "n_forward_steps": 118341,
        "loader": {
            "dataset": {"data_path": ERA5_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": REFERENCE_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-IC2-vs-era5-81yr': {
        "n_forward_steps": 118341,
        "loader": {
            "dataset": {"data_path": ERA5_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": TARGET_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-c96-vs-c24-4deg-10yr-IC0': {
        "n_forward_steps": 14600,
        "loader": {
            "dataset": {"data_path": C24_4DEG_IC0_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": C96_4DEG_IC0_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-c96-vs-c24-4deg-10yr-IC1': {
        "n_forward_steps": 14600,
        "loader": {
            "dataset": {"data_path": C24_4DEG_IC1_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": C96_4DEG_IC1_DATASET_PATH},
            "start_indices": {"times": ["2001-01-01T00:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-c96-vs-c24-4deg-81yr-IC0': {
        "n_forward_steps": 118341,
        "loader": {
            "dataset": {"data_path": C24_4DEG_IC0_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": C96_4DEG_IC0_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
    'shield-amip-c96-vs-c24-4deg-81yr-IC1': {
        "n_forward_steps": 118341,
        "loader": {
            "dataset": {"data_path": C24_4DEG_IC1_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
        },
        "prediction_loader": {
            "dataset": {"data_path": C96_4DEG_IC1_DATASET_PATH},
            "start_indices": {"times": ["1940-01-01T12:00:00"]},
            "num_data_workers": 8,
        },
        "aggregator": {"monthly_reference_data": None}
    },
}

# non-best inference checkpoint runs
RANDOM_SEED_OVERLAYS = {
    "shield-amip-1deg-ace2-inference-5yr-val-RS0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1996-01-01T00:00:00",
                    "1996-02-01T00:00:00",
                    "1996-03-01T00:00:00",
                    "1996-04-01T00:00:00",
                    "1996-05-01T00:00:00",
                    "1996-06-01T00:00:00",
                    "1996-07-01T00:00:00",
                    "1996-08-01T00:00:00",
                    "1996-09-01T00:00:00",
                    "1996-10-01T00:00:00",
                    "1996-11-01T00:00:00",
                    "1996-12-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS0-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS0-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-02T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS0-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-03T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-RS0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS0-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-01T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS0-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-02T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS0-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-03T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-5yr-val-RS1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1996-01-01T00:00:00",
                    "1996-02-01T00:00:00",
                    "1996-03-01T00:00:00",
                    "1996-04-01T00:00:00",
                    "1996-05-01T00:00:00",
                    "1996-06-01T00:00:00",
                    "1996-07-01T00:00:00",
                    "1996-08-01T00:00:00",
                    "1996-09-01T00:00:00",
                    "1996-10-01T00:00:00",
                    "1996-11-01T00:00:00",
                    "1996-12-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS1-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS1-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-02T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS1-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-03T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-RS1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS1-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-01T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS1-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-02T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS1-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-03T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-5yr-val-RS2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1996-01-01T00:00:00",
                    "1996-02-01T00:00:00",
                    "1996-03-01T00:00:00",
                    "1996-04-01T00:00:00",
                    "1996-05-01T00:00:00",
                    "1996-06-01T00:00:00",
                    "1996-07-01T00:00:00",
                    "1996-08-01T00:00:00",
                    "1996-09-01T00:00:00",
                    "1996-10-01T00:00:00",
                    "1996-11-01T00:00:00",
                    "1996-12-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS2-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS2-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-02T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS2-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-03T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-RS2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS2-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-01T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS2-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-02T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS2-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS2-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-03T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-5yr-val-RS3": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1996-01-01T00:00:00",
                    "1996-02-01T00:00:00",
                    "1996-03-01T00:00:00",
                    "1996-04-01T00:00:00",
                    "1996-05-01T00:00:00",
                    "1996-06-01T00:00:00",
                    "1996-07-01T00:00:00",
                    "1996-08-01T00:00:00",
                    "1996-09-01T00:00:00",
                    "1996-10-01T00:00:00",
                    "1996-11-01T00:00:00",
                    "1996-12-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS3-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS3-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-02T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-10yr-RS3-IC2": (
        "01J5Y2WQ5ZV3WZXBMZP32BG81N",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-03T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-RS3": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS3-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-01T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS3-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-02T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace2-inference-81yr-RS3-IC2": (
        "brianhenn/shield-amip-1deg-ace2-train-RS3-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-03T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace-inference-81yr-noCO2-RS1-IC0": (
        "brianhenn/shield-amip-1deg-ace2-train-noCO2-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-01T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace-inference-81yr-noCO2-RS1-IC1": (
        "brianhenn/shield-amip-1deg-ace2-train-noCO2-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-02T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace-inference-81yr-noCO2-RS1-IC3": (
        "brianhenn/shield-amip-1deg-ace2-train-noCO2-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-03T12:00:00"]},
            },
            "data_writer": {
                "save_monthly_files": True,
                "names": [
                    "TMP2m",
                    "SHTFLsfc",
                    "global_mean_co2",
                ],
            }
        },
    ),
    "shield-amip-1deg-ace-climsst-inference-10yr-IC0": (
        "brianhenn/ace-checkpoint-2023",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-01T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace-climsst-inference-10yr-IC1": (
        "brianhenn/ace-checkpoint-2023",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-02T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace-climsst-inference-10yr-IC2": (
        "brianhenn/ace-checkpoint-2023",
        {
            "n_forward_steps": 14600,
            "loader": {
                "start_indices": {"times": ["2001-01-03T00:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace-climsst-inference-81yr-IC0": (
        "brianhenn/ace-checkpoint-2023",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-01T12:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace-climsst-inference-81yr-IC1": (
        "brianhenn/ace-checkpoint-2023",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-02T12:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace-climsst-inference-81yr-IC2": (
        "brianhenn/ace-checkpoint-2023",
        {
            "n_forward_steps": 118341,
            "loader": {
                "start_indices": {"times": ["1940-01-03T12:00:00"]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-no-mois-cons-RS0": (
        "brianhenn/shield-amip-1deg-ace2-train-no-mois-cons-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-no-mois-cons-RS1": (
        "brianhenn/shield-amip-1deg-ace2-train-no-mois-cons-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-no-mois-dryair-cons-RS0": (
        "brianhenn/shield-amip-1deg-ace2-train-no-mois-dryair-cons-RS0-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
    "shield-amip-1deg-ace2-inference-60yr-no-mois-dryair-cons-RS1": (
        "brianhenn/shield-amip-1deg-ace2-train-no-mois-dryair-cons-RS1-best-inference-ckpt",
        {
            "n_forward_steps": 7300,
            "forward_steps_in_memory": 5,
            "loader": {
                "start_indices": {"times": [
                    "1940-01-01T00:00:00",
                    "1945-01-01T00:00:00",
                    "1950-01-01T00:00:00",
                    "1955-01-01T00:00:00",
                    "1960-01-01T00:00:00",
                    "1965-01-01T00:00:00",
                    "1970-01-01T00:00:00",
                    "1975-01-01T00:00:00",
                    "1980-01-01T00:00:00",
                    "1985-01-01T00:00:00",
                    "1990-01-01T00:00:00",
                    "1995-01-01T00:00:00",
                ]},
            },
        },
    ),
}

def merge_configs(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested configurations."""
    base_copy = base.copy() # don't modify the original base
    for k, v in new.items():
        if isinstance(v, dict):
            base_copy[k] = merge_configs(base_copy.get(k, {}), v)
        else:
            base_copy[k] = v
    return base_copy


def write_config_dataset(config: Dict[str, Any]):
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, DATASET_CONFIG_FILENAME)
        with open(filepath, "w") as f:
            yaml.safe_dump(config, f)
        dataset_name = "ace-inference-config-" + str(uuid.uuid4())[:8]
        dataset = client.dataset.create(dataset_name, filepath)
    return dataset


def get_experiment_spec(
    name: str,
    config: Dict[str, Any],
    image_name=IMAGE_NAME,
    trained_model_dataset_id=TRAINED_MODEL_DATASET_ID,
):
    """Given a dict representing the inference configuration, return a beaker experiment spec."""
    config_dataset = write_config_dataset(config)
    env_vars = [
        beaker.EnvVar(name="WANDB_API_KEY", secret="wandb-api-key-ai2cm-sa"),
        beaker.EnvVar(name="WANDB_JOB_TYPE", value="inference"),
        beaker.EnvVar(name="WANDB_NAME", value=name),
        beaker.EnvVar(name="WANDB_RUN_GROUP", value="shield-amip-ace2-inference"),
        beaker.EnvVar(name="WANDB_USERNAME", value="bhenn1983")
    ]
    datasets = [
        beaker.DataMount(
            source=beaker.DataSource(beaker=config_dataset.id),
            mount_path=DATASET_CONFIG_MOUNTPATH,
        ),
        beaker.DataMount(
            mount_path="/ckpt.tar",
            source=beaker.DataSource(beaker=trained_model_dataset_id),
            sub_path=f"training_checkpoints/{CHECKPOINT_NAME}",
        ),
        beaker.DataMount(
            mount_path="/climate-default",
            source=beaker.DataSource(weka="climate-default"),
        ),
    ]
    spec = beaker.ExperimentSpec(
        budget="ai2/climate",
        description="Do inference with ACE2 model trained on SHiELD-AMIP.",
        tasks=[
            beaker.TaskSpec(
                name=name,
                image=beaker.ImageSource(beaker=image_name),
                command=[
                    "python",
                    "-m",
                    "fme.ace.evaluator",
                    f"{DATASET_CONFIG_MOUNTPATH}/{DATASET_CONFIG_FILENAME}",
                ],
                result=beaker.ResultSpec(path="/output"),
                resources=beaker.TaskResources(gpu_count=1, shared_memory="50GiB"),
                context=beaker.TaskContext(priority="high", preemptible=True),
                constraints=beaker.Constraints(cluster=["ai2/jupiter-cirrascale-2"]),
                env_vars=env_vars,
                datasets=datasets,
            )
        ],
    )
    return spec


if __name__ == "__main__":
    client = beaker.Beaker.from_env()

    with open(LOCAL_BASE_CONFIG_FILENAME, "r") as f:
        base_config = yaml.safe_load(f)

    print("Validating that configs have correct types.")
    for name, overlay in EXPERIMENT_OVERLAYS.items():
        config = merge_configs(base_config, overlay)
        print(f"Validating config for experiment {name}.")
        print(f"Config that is being validated:\n{config}")
        dacite.from_dict(
            fme.ace.InferenceEvaluatorConfig, config, config=dacite.Config(strict=True)
        )
    print("All configs are valid. Starting experiment submission.")
    for name, overlay in EXPERIMENT_OVERLAYS.items():
        config = merge_configs(base_config, overlay)
        print(f"Creating experiment {name}.")
        spec = get_experiment_spec(name, config)
        try:
            experiment = client.experiment.create(name, spec)
            print(
                f"Experiment {name} created. See https://beaker.org/ex/{experiment.id}"
            )
        except beaker.exceptions.ExperimentConflict:
            print(
                f"Failed to create experiment {name} because it already exists. "
                "Skipping experiment creation. If you want to submit this experiment, "
                "delete the existing experiment with the same name, or rename the new "
                "experiment."
            )

    for name, (checkpoint, overlay) in RANDOM_SEED_OVERLAYS.items():
        config = merge_configs(base_config, overlay)
        print(f"Validating config for experiment {name}.")
        print(f"Config that is being validated:\n{config}")
        try:
            dacite.from_dict(
                fme.ace.InferenceEvaluatorConfig, config, config=dacite.Config(strict=True)
            )
        except:
            print(f"Error in config for experiment {name}.")
            break
        print(f"Creating experiment {name}.")
        spec = get_experiment_spec(name, config, trained_model_dataset_id=checkpoint)
        try:
            experiment = client.experiment.create(name, spec)
            print(
                f"Experiment {name} created. See https://beaker.org/ex/{experiment.id}"
            )
        except beaker.exceptions.ExperimentConflict:
            print(
                f"Failed to create experiment {name} because it already exists. "
                "Skipping experiment creation. If you want to submit this experiment, "
                "delete the existing experiment with the same name, or rename the new "
                "experiment."
            )
