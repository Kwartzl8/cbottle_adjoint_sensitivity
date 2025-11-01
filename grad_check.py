import argparse
import os
import torch
import xarray as xr
import asyncio
import datetime
import matplotlib.pyplot as plt

import earth2grid.healpix as healpix
from cbottle.datasets.amip_sst_loader import AmipSSTLoader

def concat_dataset_and_extract_q_names(output_folder: str, additional_vars: list[str] = []) -> tuple[xr.Dataset, list[str]]:
    netcdf_files = [f for f in os.listdir(output_folder) if f.endswith('.nc')]
    if not netcdf_files:
        raise ValueError(f"No netcdf files found in {output_folder}")
    
    # Automatically get variable names from the first netcdf file.
    dataset_0 = xr.open_dataset(os.path.join(output_folder, netcdf_files[0]))
    print(f"Variables in the dataset: {list(dataset_0.data_vars.keys())}")
    field_grad_variables = []
    scalar_grad_variables = []
    q_names = additional_vars.copy()
    for var_name in dataset_0.data_vars.keys():
        if '_grad' in var_name:
            q_name = var_name[:str.find(var_name, '_grad')]
            if 'grid_mapping' in dataset_0[var_name].attrs:
                field_grad_variables.append(var_name)
            else:
                scalar_grad_variables.append(var_name)
            q_names.append(q_name)
    # There are repeated q_names if both field and scalar gradients exist. Get unique names.
    q_names = list(set(q_names))

    # Slice the netcdf files to only get the q_values and the grad vars, and then concatenate them.
    for nc_file in netcdf_files:
        file_path = os.path.join(output_folder, nc_file)
        dataset = xr.open_dataset(file_path)
        # TODO also select the conditioning SST that was used when it cannot be retrieved from AMIP with just the timestamps.
        selected_vars = q_names + field_grad_variables + scalar_grad_variables
        dataset_sliced = dataset[selected_vars]
        if nc_file == netcdf_files[0]:
            combined_dataset = dataset_sliced
        else:
            combined_dataset = xr.concat([combined_dataset, dataset_sliced], dim='time')

    return combined_dataset, q_names


async def load_conditioning_sst(times_to_select):
    grid = healpix.Grid(
        level=6, pixel_order=healpix.PixelOrder.NEST
    )
    amip_loader = AmipSSTLoader(grid)
    sst = await amip_loader.sel_time(times_to_select)
    sst = healpix.reorder(torch.as_tensor(sst[('tosbcs', -1)]), healpix.PixelOrder.NEST, healpix.PixelOrder.RING)
    return sst.float()

def get_differences(ds: xr.Dataset, q_names: list[str], sst_conditioning, times) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # Extract doy and tod from times
    doys = []
    tods = []
    for time in times:
        day_start = time.replace(hour=0, minute=0, second=0)
        year_start = day_start.replace(month=1, day=1)
        day_of_year = (time - year_start) / datetime.timedelta(seconds=86400)
        second_of_day = (time - day_start) / datetime.timedelta(seconds=1)
        doys.append(day_of_year)
        tods.append(second_of_day)
    doys = torch.tensor(doys)  # shape (time,)
    tods = torch.tensor(tods)  # shape (time,)

    # We denote scalar grad variables as dq/ds, where s can be, in the case of cBottle,
    # (fractional) day of year (DoY) - from 1 to 365.25 - or time of day (ToD) - from 0 to 86400 seconds.

    # For the full derivative dq/dc, we need to combine them with the chain rule:
    # dq/dc = ∂q/∂c + ∂q/∂DoY * ∂DoY/dc + ∂q/∂ToD * ∂ToD/∂c
    # where c is the conditioning SST.
    finite_diff_sst = torch.diff(sst_conditioning, dim=0)  # shape (time-1, hpx)
    finite_diff_doy = torch.diff(doys, dim=0)  # shape (time-1,)
    finite_diff_tod = torch.diff(tods, dim=0)  # shape (time-1,)

    q_values = {}
    finite_delta_q = {}
    linearized_delta_q = {}
    for q_name in q_names:
        field_grad_var = f"{q_name}_grad_sst"
        doy_grad_var = f"{q_name}_grad_doy"
        tod_grad_var = f"{q_name}_grad_tod"

        partial_q_partial_c = torch.tensor(ds[field_grad_var].data)
        partial_q_partial_doy = torch.tensor(ds[doy_grad_var].data)
        partial_q_partial_tod = torch.tensor(ds[tod_grad_var].data)
        q_val = torch.tensor(ds[q_name].data)
        q_values[q_name] = q_val

        # Do grad-check: for every 2 consecutive datapoints, average gradients from the model and compare to the finite difference.
        avg_partial_q_partial_c = 0.5 * (partial_q_partial_c[:-1, :] + partial_q_partial_c[1:, :])
        avg_partial_q_partial_doy = 0.5 * (partial_q_partial_doy[:-1] + partial_q_partial_doy[1:])
        avg_partial_q_partial_tod = 0.5 * (partial_q_partial_tod[:-1] + partial_q_partial_tod[1:])
        
        # Combine using chain rule
        linearized_delta_q[q_name] = torch.sum(avg_partial_q_partial_c * finite_diff_sst, dim=1) + avg_partial_q_partial_doy * finite_diff_doy + avg_partial_q_partial_tod * finite_diff_tod
        finite_delta_q[q_name] = torch.diff(q_val, dim=0)

    return q_values, finite_delta_q, linearized_delta_q


def plot_grad_check(q_values: torch.Tensor, linearized_delta_q: torch.Tensor, times, q_name: str, output_folder: str):
    plt.figure(figsize=(20, 6))
    plt.scatter(range(len(q_values)), q_values.numpy(), label='q values', alpha=0.5)
    for t in range(len(linearized_delta_q)):    # only plot up to T-1 as that is how many finite diffs we have
        plt.plot([t, t+1], [q_values[t].item(), q_values[t].item() + linearized_delta_q[t].item()], color='red', alpha=0.7)
    plt.title(f"Gradient Check for {q_name}.")
    plt.xlabel("Time Index")
    num_ticks = min(20, len(times))
    tick_indices = [int(i * (len(times) - 1) / (num_ticks - 1)) for i in range(num_ticks)] if num_ticks > 1 else [0]
    plt.xticks(tick_indices)
    plt.gca().set_xticklabels([times[i].strftime('%Y-%m-%d %H') for i in tick_indices], rotation=25)
    plt.ylabel(f"{q_name} values")
    plt.legend()
    if not os.path.exists(os.path.join(output_folder, "grad_check_plots")):
        os.makedirs(os.path.join(output_folder, "grad_check_plots"))
    plt.savefig(os.path.join(output_folder, "grad_check_plots", f"grad_check_{q_name}.png"))
    plt.close()


def main():
    args = argparse.ArgumentParser(description="Gradient Check Script")
    args.add_argument("--output_folder", type=str, help="Output folder for saving results")
    parsed_args = args.parse_args()
    output_folder = parsed_args.output_folder

    concatenated_ds, q_names = concat_dataset_and_extract_q_names(output_folder)

    # TODO SST conditioning should be in the dataset as well, but for now we retrieve it from AMIP.
    times_to_select = concatenated_ds.indexes['time']
    sst_conditioning = asyncio.run(load_conditioning_sst(times_to_select)) # tensor of shape (time, hpx)
    concatenated_ds = concatenated_ds.assign({'sst_conditioning': (('time', 'pix'), sst_conditioning)})

    # Reorder with increasing time
    concatenated_ds = concatenated_ds.sortby('time')
    sst_conditioning = torch.tensor(concatenated_ds['sst_conditioning'].data)
    times = concatenated_ds.indexes['time']
    
    q_values_dict, delta_q_finite_dict, linearized_delta_q_dict = get_differences(concatenated_ds, q_names, sst_conditioning, times)

    MAE_scores = {}
    MAE_relative_scores = {}
    for q_name in q_names:
        q_values = q_values_dict[q_name]
        delta_q_finite = delta_q_finite_dict[q_name]
        linearized_delta_q = linearized_delta_q_dict[q_name]

        MAE = torch.mean(torch.abs(linearized_delta_q - delta_q_finite)).item()
        RMSE = torch.sqrt(torch.mean((linearized_delta_q - delta_q_finite)**2)).item()
        relative_MAE = MAE / torch.mean(torch.abs(delta_q_finite)).item()
        # normalize by interquartile range
        iqr_q_values = torch.quantile(q_values, 0.75) - torch.quantile(q_values, 0.25)
        relative_RMSE = RMSE / iqr_q_values.item()
        print(f"MAE {q_name}: {MAE}. Relative to the mean of delta_q_finite: {relative_MAE}")
        print(f"RMSE {q_name}: {RMSE}. Relative to the mean of delta_q_finite: {relative_RMSE}")
        MAE_scores[q_name] = MAE
        MAE_relative_scores[q_name] = relative_MAE

        # Plot.
        plot_grad_check(q_values, linearized_delta_q, times, q_name, output_folder)


if __name__ == "__main__":
    main()
