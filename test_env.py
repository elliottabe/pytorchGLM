import ray
import logging 

import numpy as np
import pandas as pd
import scipy as sp

import xarray as xr

import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray.init(
    ignore_reinit_error=True,
    logging_level=logging.ERROR,
)

print(f'np: {np.__version__}')
print(f'pd: {pd.__version__}')
print(f'xr: {xr.__version__}')
print(f'ray: {ray.__version__}')
print(f'scipy: {sp.__version__}')
print(f'torch: {torch.__version__}')

print(f'cuda enable: {torch.cuda.is_available()}')
print(f'current_device: {torch.cuda.current_device()}')
print(f'device: {torch.cuda.device(0)}')
print(f'device_count: {torch.cuda.device_count()}')
print(f'get_device_name: {torch.cuda.get_device_name(0)}')
print(f'torch device: {device}')