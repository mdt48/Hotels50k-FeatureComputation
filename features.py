import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from glob import glob
from lib.utils import as_numpy

num_classes = 162

feat_2048 = torch.zeros(size=(1, 2048))
feat_2048_whole = torch.zeros(size=(1, 2048, 38, 50))
feat_150 = torch.Tensor

