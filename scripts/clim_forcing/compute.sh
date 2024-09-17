#!/bin/bash

python compute.py \
    --start-time 1991-01-01 \
    --end-time 2020-12-31 \
    --input-url gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr \
    gs://vcm-ml-intermediate/2024-09-04-era5-1deg-8layer-forcing-clim-1991-2020.zarr
