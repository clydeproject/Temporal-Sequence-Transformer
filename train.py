import torch 
import torch.nn as nn 
from torch.optim import Adam
from model import TST

model = TST.from_config("config.json")





