import torch
import pandas as pd
import numpy as np
from utils.logger import logger

from network import Network
from Recursive import RecursiveLogit, TimeFreeRecursiveLogit
from optimize import GPUOptimizer

#input

adj_matrix = torch.load("grid_adjacency_matrix.pt")
node_features = torch.load("node_features_matrix.pt")
demand =  pd.read_csv("demand.csv", index_col = 0)
demand = torch.tensor(demand.values[0], dtype=torch.long)
demand = demand.unsqueeze(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#output
network = Network(adj_matrix, node_features)
rl = TimeFreeRecursiveLogit(network, demand, 0.9, 15, device)
df= rl.simulation([1, 1])
df.to_csv("route_data/route.csv")
