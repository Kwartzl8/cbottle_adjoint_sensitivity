
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Alex Dobra on 2025-07-07.
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
import datetime

import cbottle.distributed as dist
import earth2grid
import numpy as np
import torch
import torch.distributed
import tqdm
from cbottle.dataclass_parser import Help, a, parse_args
import cbottle.datasets.dataset_3d as dataset_3d
from cbottle.datasets.base import BatchInfo
from src.netcdf_writer import NetCDFConfig, NetCDFWriter
from cbottle.denoiser_factories import get_denoiser, DenoiserType
from src.QFactory import QFactory
from src.adjoint_samplers import edm_reverse_sampler_with_conditioning_gradients, get_conditional_denoiser
from cbottle.diffusion_samplers import (
    edm_sampler_from_sigma,
)
from cbottle.datasets.merged_dataset import TimeMergedDataset
from src.config_loader import load_config

logger = logging.getLogger(__name__)


def prepare_for_saving(
    x: torch.Tensor, hpx: earth2grid.healpix.Grid, batch_info: BatchInfo
):
    """
    Denormalizes and reorders data to RING order
    """
    x = batch_info.denormalize(x)
    ring_order = hpx.reorder(earth2grid.healpix.PixelOrder.RING, x)
    return {batch_info.channels[c]: ring_order[:, c] for c in range(x.shape[1])}


def generate_clean_image(net, latents, condition_data, cfg):
    """Runs the forward diffusion process to generate a clean image."""
    D = get_denoiser(
        net=net,
        images=latents,
        labels=condition_data['labels'],
        condition=condition_data['condition'],
        second_of_day=condition_data['second_of_day'],
        day_of_year=condition_data['day_of_year'],
        denoiser_type=DenoiserType.standard,
        sigma_max=cfg.sample.sigma_max,
    )

    with torch.autocast("cuda", enabled=cfg.sample.bf16, dtype=torch.bfloat16):
        clean_image = edm_sampler_from_sigma(
            D,
            latents,
            randn_like=torch.randn_like,
            sigma_max=int(cfg.sample.sigma_max),
            num_steps=cfg.sample.num_steps,
        )
    return clean_image


def compute_gradients(net, clean_image, condition_data, aggregation_fns, cfg):
    """Runs the reverse adjoint sampling process to extract gradients."""
    cond_D = get_conditional_denoiser(
        net=net,
        condition=condition_data['condition'],
        day_of_year=condition_data['day_of_year'],
        second_of_day=condition_data['second_of_day'],
        labels=condition_data['labels'],
    )
    
    generated_noise, qs, dqdc, dqddoy, dqdtod = edm_reverse_sampler_with_conditioning_gradients(
        net=cond_D,
        clean_image=clean_image,
        day_of_year=condition_data['day_of_year'],
        second_of_day=condition_data['second_of_day'],
        condition=condition_data['condition'],
        aggregation_fns=aggregation_fns,
        num_steps=cfg.sample.num_steps,
    )
    return dqdc, qs, dqddoy, dqdtod


def setup_q_functions(batch_info: BatchInfo, hpx_level: int, q_names: list[str]):
    q_factory = QFactory(batch_info, hpx_level=hpx_level)
    q_function_map = {
        "global_TOA_radiation": q_factory.get_global_TOA_outgoing_radiation().aggr_func,
        "global_rlut": q_factory.get_global_avg_in_channel('rlut').aggr_func,
        "global_rsut": q_factory.get_global_avg_in_channel('rsut').aggr_func,
        "indian_ocean_rsut": q_factory.get_patch_avg_in_channel('rsut', min_lat=-20, max_lat=10, min_long=60, max_long=90).aggr_func,
        "indian_ocean_sst": q_factory.get_patch_avg_in_channel('sst', min_lat=-20, max_lat=-10, min_long=60, max_long=70).aggr_func,
        "global_precipitation": q_factory.get_global_avg_in_channel('pr').aggr_func,
        "global_surface_temperature": q_factory.get_global_avg_in_channel('tas').aggr_func,
        # "upward_pressure_velocity_500hPa": q_factory.get_pressure_velocity_500hPa().aggr_func,
        "global_water_vapor": q_factory.get_global_avg_in_channel('tcwv').aggr_func,
        "global_sic": q_factory.get_global_avg_in_channel('sic').aggr_func,
        "north_atlantic_rlut": q_factory.get_patch_avg_in_channel('rlut', min_lat=45, max_lat=55, min_long=-30, max_long=-20).aggr_func,
        "north_atlantic_rsut": q_factory.get_patch_avg_in_channel('rsut', min_lat=45, max_lat=55, min_long=-30, max_long=-20).aggr_func,
    }

    aggr_fns = []
    valid_q_names = []
    gradient_scaling = []
    for q_name in q_names:
        if q_name not in q_function_map:
            logger.warning(f"Unknown q_name {q_name}. Known names: {list(q_function_map.keys())}")
            continue
        valid_q_names.append(q_name)
        aggr_fns.append(q_function_map[q_name])
        gradient_scaling.append(1 / q_factory.diff_var_scale)
    
    if len(aggr_fns) == 0:
        raise ValueError("No valid q_names provided.")
    return valid_q_names, aggr_fns, gradient_scaling


def get_dataset(cfg, rank, world_size, climatology_sst) -> TimeMergedDataset:
    import cbottle.datasets.dataset_3d as dataset_3d
    if not climatology_sst:
        new_amip_metadata = dataset_3d.DatasetMetadata(
            name="amip",
            start=cfg.start_date,
            end=cfg.end_date,
            time_step=cfg.time_step_hours,
            time_unit=dataset_3d.TimeUnit.HOUR,
        )
        dataset_3d.DATASET_METADATA['amip'] = new_amip_metadata
    else:
        from src.make_climatology_sst_dataset import make_climatology_sst_dataset, ARBITRARY_YEAR
        # Create climatology SST dataset if not already present
        climatology_amip_path = 'amip_sst_climatology.nc'
        # Check if file exists
        if not os.path.isfile(climatology_amip_path):
            make_climatology_sst_dataset(climatology_amip_path)
        dataset_3d.AmipSSTLoader.path = climatology_amip_path

        # Replace the year from the config with our arbitrary year for climatology
        start_date = datetime.datetime.strptime(cfg.start_date, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.datetime.strptime(cfg.end_date, "%Y-%m-%d %H:%M:%S")
        start_date = start_date.replace(year=ARBITRARY_YEAR)
        end_date = end_date.replace(year=ARBITRARY_YEAR)
        # check if end_date is before start_date (e.g., Dec 31 to Jan 2)
        if end_date < start_date:
            # flip them
            start_date, end_date = end_date, start_date

        climatology_amip_metadata = dataset_3d.DatasetMetadata(
            name="amip_climatology",
            start=start_date,
            end=end_date,
            time_step=cfg.time_step_hours,
            time_unit=dataset_3d.TimeUnit.HOUR,
        )
        dataset_3d.DATASET_METADATA['amip'] = climatology_amip_metadata

    dataset = dataset_3d.get_dataset(
        rank=rank,
        world_size=world_size,
        split=cfg.data_split,
        dataset=cfg.dataset.name,
        sst_input=True,
        infinite=False,
        shuffle=False,
        chunk_size=3
    )
    dataset.infinite = False
        
    return dataset


def set_up_distributed_training() -> tuple[int, int]:
    # TEMPORARY for debugging on single GPU when rank=0 is busy
    # if torch.cuda.device_count() > 1:
    #     os.environ["LOCAL_RANK"] = str(3)
    #     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
    dist.init()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # get slurm vars if present
    id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))  # 1-indexed
    slurm_array_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    rank = rank + world_size * (id - 1)
    world_size = world_size * slurm_array_count
    return rank, world_size


def main():
    logging.basicConfig(level=logging.INFO)
    args = load_config(
        # "configs/1971-2020_TOA_rad_config.yaml"
        "configs/base_config.yaml"
        # "configs/debug_config_backprop.yaml"
        # "configs/2020_north_atlantic_rlut_config_96h.yaml"
    )
    rank, world_size = set_up_distributed_training()

    # TEMPORARY Delete the contents of inference_output/trash.
    if rank == 0 and os.path.exists(args.output_path) and 'trash' in args.output_path:
        for file in os.listdir(args.output_path):
            file_path = os.path.join(args.output_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    # Load model
    from cbottle.checkpointing import Checkpoint
    with Checkpoint(args.state_path) as checkpoint:
        net = checkpoint.read_model()
    net.eval()
    net.requires_grad_(False)
    net.float()
    net.cuda()

    # Load data
    dataset = get_dataset(args, rank, world_size, args.sample.climatology_sst)
    batch_info = dataset.batch_info

    q_names, aggr_fns, gradient_scaling = setup_q_functions(batch_info, hpx_level=args.hpx_level, q_names=args.sample.q_names)
    gradient_centers = [0.0] * len(aggr_fns)
    # Extract batch_info and modify it for saving gradients TODO do all this in setup_q_functions.
    gradient_channels = [q_name + "_grad_sst" for q_name in q_names]
    if args.sample.save_data:
        batch_info.channels = batch_info.channels + gradient_channels
        batch_info.scales = np.concatenate([batch_info.scales, gradient_scaling])
        batch_info.center = np.concatenate([batch_info.center, gradient_centers])
    else:
        batch_info.channels = gradient_channels
        batch_info.scales = gradient_scaling
        batch_info.center = gradient_centers
        
    # Initialize netCDF writer
    nc_config = NetCDFConfig(
        hpx_level=args.hpx_level,
        time_units=dataset.time_units,
        calendar=dataset.calendar,
        attrs=None,
    )
    writer = NetCDFWriter(args.output_path, nc_config, batch_info.channels, rank=rank)
    writer._add_scalar_variables(args.sample.q_names)
    writer._add_scalar_variables([q_name + "_grad_doy" for q_name in q_names])
    writer._add_scalar_variables([q_name + "_grad_tod" for q_name in q_names])

    # If min_samples == -1, process all samples.
    if args.sample.min_samples > 0:
        dataset.set_times(dataset.times[: args.sample.min_samples])

    # Skip times that have already been processed
    try:
        logger.info(
            f"Skipping {writer.time_index} times out of {len(dataset._times)}"
        )
        dataset._times = dataset._times[writer.time_index :]
    except AttributeError:
        pass

    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.sample.batch_gpu, sampler=None, drop_last=False
    )

    # Set fixed seed for reproducibility
    if args.sample.seed is not None:
        torch.manual_seed(args.sample.seed)
        torch.cuda.manual_seed_all(args.sample.seed)

    # Initialize a fixed noise tensor if required.
    fixed_latents = None
    if args.sample.use_fixed_noise:
        # Define shape based on model and batch size
        latent_shape = (1, net.img_channels, net.time_length, net.domain.numel())
        fixed_latents = torch.randn(latent_shape, device='cuda')
        torch.save(fixed_latents, f"noise_backprop_seed{args.sample.seed}.pt")
    
    for batch in tqdm.tqdm(loader, disable=rank != 0):
        if (args.sample.min_samples > 0) and (
            writer.time_index * world_size > args.sample.min_samples
        ):
            break
        # Reshape latent noise for last batch if smaller than batch size.
        condition_data = {
            'condition': batch["condition"].cuda(),
            'labels': batch["labels"].cuda(),
            'second_of_day': batch["second_of_day"].cuda().float(),
            'day_of_year': batch["day_of_year"].cuda().float(),
        }

        if args.sample.use_fixed_noise:
            latents = fixed_latents.repeat(batch['target'].shape[0], 1, 1, 1)
        else:
            latents = torch.randn_like(batch['target'], device='cuda')
        clean_image = generate_clean_image(net, latents, condition_data, args)
        gradients, q_list, dqddoy, dqdtod = compute_gradients(net, clean_image, condition_data, aggr_fns, args)

        scalar_dict = {q_name: q.detach().cpu() for q_name, q in zip(q_names, q_list)}
        # Append dqdtau values to the scalar dict
        for i, q_name in enumerate(q_names):
            scalar_dict[q_name + "_grad_doy"] = dqddoy[:, i].detach().cpu()
            scalar_dict[q_name + "_grad_tod"] = dqdtod[:, i].detach().cpu()

        if args.sample.save_data:
            out = torch.cat([clean_image, gradients], dim=1)
        else:
            out = gradients
        ring_denormalized_data = prepare_for_saving(
            out, net.domain._grid, batch_info
        )

        # Convert time data to timestamps
        timestamps = batch["timestamp"]
        writer.write_batch(
            ring_denormalized_data,
            timestamps,
            scalars=scalar_dict,
        )

if __name__ == "__main__":
    main()
