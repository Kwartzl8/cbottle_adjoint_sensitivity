[![arXiv](https://img.shields.io/badge/arXiv-2511.00663-b31b1b.svg)](https://arxiv.org/abs/2511.00663)
# Sensitivity Analysis with Climate in a Bottle
![alt text](grad_check_north_atlantic_rsut_2020-01-01_2020-03-05.png)
Accepted at CCAI-NeurIPS25. [ArXiv link](https://arxiv.org/abs/2511.00663).
# Installation

## Download cBottle weights
First, download the coarse Climate in a Bottle model weights with:
```
curl -L 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/earth-2/cbottle/1.2/files?redirect=true&path=cBottle-3d/training-state-009856000.checkpoint' -o 'training-state-009856000.checkpoint'
```

In the configs the checkpoint file is referred to as `cBottle-3d.zip` so rename it to that:
```
mv training-state-009856000.checkpoint cBottle-3d.zip
```

## Install dependencies
Activate your virtual environment and then install from `requirements.txt`:
```
pip install -r requirements.txt
```

# Running
## Run adjoint sensitivity algorithm
To run on a single GPU using `basic_config.yaml`:

```
python backprop_gradients_amip_sst.py
```

Run on multiple GPUs:
```
torchrun --nproc-per-node=2 backprop_gradients_amip_sst.py
```

## Run grad-check
```
python grad_check --output_folder OUTPUT_FOLDER
```
This produces a plot showing the function q(SST) and its gradients (in the direction of SST changing with time).

# Reproducing plots from the paper
## Global net radiation sensitivity with respect to SST
In the `main()` function in `backprop_gradients_amip_sst.py` load the config `configs/1971-2020_TOA_rad_config.yaml`. After setting an appropriate batch size and number of discretization steps, run as shown above. This took about 8-10h on an A100 with the current settings.
To see the plots and get the RMSE score, run the two notebooks with the appropriate paths and quantity names.
## Short-wave radiation in north Atlantic sensitivity with respect to SST
Now load the config `2020_north_atlantic_rsut_config.yaml` and run the scripts the same way.
## Adjoint method illustration
Run the notebook `plot_adjoint_method_illustration.ipynb`.

# Extending to other quantities of interest

The adjoint sensitivity algorithm can be applied to obtain sensitivities of any differentiable function $q(X_0)$ of cBottle's outputs. To do this, follow the steps:
1. Go to `src/QFactory.py` and define a new `get_my_q()->QStruct` function within the class. More general functions for getting averages within a given lat-lon box or globally are already defined. Note: I am not denormalizing the sensitivities obtained (except with 1/`SST_SCALE`, see `setup_q_functions` in `backprop_gradients_amip_sst.py`), and so this needs to be done within the `get_my_q()` function.
2. Add the pair `q_name` of your choice and `get_my_q()` to the dictionary in `setup_q_functions` in `backprop_gradients_amip_sst.py` to be able to use `q_name` directly in the config.
3. Create new config file with the same format as `basic_config.yaml` and include your `q_name`. Make sure to load this in the `main()` function in `backprop_gradients_amip_sst.py`. For further customization, `config_loader.py` might also need to be changed to suit your needs.

