import torch
import torch.nn as nn

import pickle
from decimal import Decimal, getcontext
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys



N=2

t=1

J=16*t
mu=-8.3*t
T=0.1*t

