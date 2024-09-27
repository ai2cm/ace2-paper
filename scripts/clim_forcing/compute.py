"""Purpose of this script is to generate a forcing file to allow running the
ACE2-ERA5 model with climatological boundary conditions."""

import argparse
import numpy as np
from dask.diagnostics import ProgressBar
import xarray as xr

DEFAULT_START_TIME = "1990-01-01"
DEFAULT_END_TIME = "2020-12-31"
FORCING_VARIABLES = [
    "surface_temperature",
    "DSWRFtoa",
    "land_fraction",
    "ocean_fraction",
    "sea_ice_fraction",
    "global_mean_co2",
    "HGTsfc"
]
DEFAULT_URL_ERA5 = "gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr"

def main(input_url, start_time, end_time, output_url):
    ds = xr.open_zarr(input_url)
    ds = ds.sel(time=slice(start_time, end_time))
    ds = ds[FORCING_VARIABLES]

    # want a 4-year interval so that runs which repeat this dataset still have
    # a leap day once every 4 years
    output_start_time = "2001-01-01"
    output_end_time = "2004-12-31"

    clim_ds = xr.Dataset()

    # special handling for some variables
    clim_ds["DSWRFtoa"] = ds["DSWRFtoa"].sel(time=slice(output_start_time, output_end_time))
    clim_ds["global_mean_co2"] = ds["global_mean_co2"].mean("time")
    clim_ds["HGTsfc"] = ds["HGTsfc"]
    clim_ds["land_fraction"] = ds["land_fraction"]
    remaining_variables = sorted(list(set(FORCING_VARIABLES) - set(clim_ds)))
    
    # for other variables, compute monthly climatology and then interpolate
    # back to 6-hourly frequency spanning 4 years
    for name in remaining_variables:
        monthly_clim = ds[name].groupby("time.month").mean("time")
        months = monthly_clim["month"].values
        repeated_monthly_clim = []
        for year in range(2000, 2006):
            current_year = monthly_clim
            current_year = current_year.rename({"month": "time"})
            current_year['time'] = [
                np.datetime64(f'{year:04d}-{month:02d}')
                for month in months
            ]
            repeated_monthly_clim.append(current_year)
        repeated_monthly_clim = xr.concat(repeated_monthly_clim, dim="time")
        clim_ds[name] = repeated_monthly_clim.interp(time=clim_ds.time)

    clim_ds = clim_ds.chunk({"time": 40})

    with ProgressBar():
        clim_ds.to_zarr(output_url, mode='w')

def get_parser():
    """Return parser that gets a start time, end time, input URL and output URL."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-time', default=DEFAULT_START_TIME, type=str)
    parser.add_argument('--end-time', default=DEFAULT_END_TIME, type=str)
    parser.add_argument('--input-url', default=DEFAULT_URL_ERA5, type=str)
    parser.add_argument('output', type=str)
    return parser
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.input_url, args.start_time, args.end_time, args.output)