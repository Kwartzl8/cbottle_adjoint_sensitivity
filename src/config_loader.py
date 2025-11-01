"""
Configuration loader for loading YAML config files and converting them to CLI dataclass arguments.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from enum import Enum
from dataclasses import dataclass
from enum import Enum, auto
from cbottle.denoiser_factories import DenoiserType


class Sampler(Enum):
    all = auto()


class Dataset(Enum):
    era5 = auto()
    amip = auto()


@dataclass(frozen=True)
class SamplerArgs:
    q_names: list[str] | None = None
    min_samples: int = 1
    batch_gpu: int = 1
    start_from_clean_image: bool = False
    climatology_sst: bool = False
    sigma_max: float = 80.0
    num_steps: int = 18
    save_data: bool = False
    bf16: bool = False
    seed: int | None = None
    use_fixed_noise: bool = False


@dataclass
class CLI:
    state_path: str
    output_path: str
    dataset: Dataset = Dataset.amip
    start_date: str = ""
    end_date: str = ""
    time_step_hours: int = 24
    data_split: str = ""
    sample: SamplerArgs = SamplerArgs()
    hpx_level: int = 6


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the parsed configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def config_to_cli_args(config: Dict[str, Any], CLI, SamplerArgs, Dataset, Sampler, DenoiserType):
    """
    Convert a configuration dictionary to CLI dataclass arguments.
    
    Args:
        config: Configuration dictionary from YAML file
        CLI: CLI dataclass type
        SamplerArgs: SamplerArgs dataclass type
        Dataset: Dataset enum type
        Sampler: Sampler enum type
        DenoiserType: DenoiserType enum type
        
    Returns:
        Populated CLI dataclass instance
    """
    
    # Extract dataset enum value
    dataset_name = config.get('dataset', {}).get('name', 'amip')
    dataset_enum = Dataset[dataset_name]
    
    # Extract sampling configuration
    sampling_config = config.get('sampling', {})
    
    # Determine seed value (handle None/null)
    seed_value = config.get('experiment', {}).get('seed', None)
    
    # Create SamplerArgs
    sampler_args = SamplerArgs(
        q_names=config.get('q_function', {}).get("names", []),
        min_samples=sampling_config.get('min_samples', 1),
        batch_gpu=sampling_config.get('batch_gpu', 1),
        start_from_clean_image=sampling_config.get('start_from_clean_image', False),
        climatology_sst=sampling_config.get('climatology_sst', False),
        sigma_max=sampling_config.get('sigma_max', 80.0),
        num_steps=sampling_config.get('num_steps', 18),
        save_data=config.get('saving', {}).get('save_generated_data', False),
        bf16=sampling_config.get('bf16', False),
        seed=seed_value,
        use_fixed_noise=sampling_config.get('use_fixed_noise', False),
    )
    
    # Create CLI args
    output_path = config.get('experiment', {}).get('output_path', 'inference_output/trash')
    # Remove leading slash if present
    if output_path.startswith('/'):
        output_path = output_path[1:]
    
    cli_args = CLI(
        state_path=config.get('paths', {}).get('state_path', 'cBottle-3d.zip'),
        output_path=output_path,
        dataset=dataset_enum,
        start_date=config.get('dataset', {}).get('start_date', ''),
        end_date=config.get('dataset', {}).get('end_date', ''),
        time_step_hours=config.get('dataset', {}).get('time_step_hours', 24),
        data_split=config.get('dataset', {}).get('split', ''),
        sample=sampler_args,
        hpx_level=config.get('saving', {}).get('hpx_level', 6),
    )
    
    return cli_args


def load_config(config_path: str):
    """
    Load a YAML configuration file and convert it to CLI dataclass arguments.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Populated CLI dataclass instance
    """
    config = _load_yaml_config(config_path)
    cli_config =  config_to_cli_args(config, CLI, SamplerArgs, Dataset, Sampler, DenoiserType)

    # Do some sanity checks.
    if cli_config.sample.start_from_clean_image:
        assert cli_config.dataset == Dataset.era5, "Only ERA5 dataset supports start_from_clean_image=True"
    else:
        assert cli_config.dataset == Dataset.amip, "No point in loading ERA5 data if we're generating it."

    return cli_config
