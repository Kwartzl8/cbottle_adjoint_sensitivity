import os
import xarray as xr
from cbottle.config.environment import CACHE_DIR

# Set an arbitrary year for the climatology time coordinate assignment
ARBITRARY_YEAR = 2001
def make_climatology_sst_dataset(output_file):
    input_dir = os.path.join(
        CACHE_DIR,
        "tosbcs_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-9_gn_187001-202212.nc",
    )
    ds = xr.open_dataset(input_dir, engine="h5netcdf").load()
    # Average over years 1971-2020 for each month.
    times = ds.indexes["time"]
    unique_month_times = times[times.year==ARBITRARY_YEAR]

    # Average over years 1971-2020 for each month.
    climatology = []
    for month in range(1, 13):
        monthly_data = ds["tosbcs"].sel(
            time=times[(times.year >= 1971) & (times.year <= 2020) & (times.month == month)]
        )
        monthly_mean = monthly_data.mean(dim="time")
        climatology.append(monthly_mean)

    # Combine into a new dataset, with the same time coordinate but the year set to the arbitrary year.
    climatology_sst_ds = xr.concat(climatology, dim="time").assign_coords(
        time=unique_month_times
    )
    # save to netCDF file
    climatology_sst_ds.to_netcdf(output_file)

if __name__ == "__main__":
    output_climatology_file = "amip_sst_climatology.nc"
    make_climatology_sst_dataset(
        output_climatology_file,
    )
    print(f"Climatology SST dataset saved to {output_climatology_file}")