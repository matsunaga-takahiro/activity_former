import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from network import Network
from Recursive import RecursiveLogit
from optimize import GPUOptimizer

from utils.logger import logger

#base_path = os.path.dirname(os.path.abspath(__file__))

start_time = time.time()

adj_matrix = torch.load("grid_adjacency_matrix.pt")
node_features = torch.load("node_features_matrix.pt")
demand =  pd.read_csv("demand.csv", index_col = 0)
trip_df = pd.read_csv("route_data/route.csv", index_col = 0)
demand = torch.tensor(demand.values, dtype=torch.long)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x_dim = 2
discount_rate = 0.9
x00 = np.zeros(x_dim) 
timestep_limit = 10
bounds = [(-500, 500) for _ in range(x_dim)]
network = Network(adj_matrix, node_features)
rl = RecursiveLogit(network, demand, discount_rate, timestep_limit, device)
optimizer = GPUOptimizer(network, trip_df, rl, bounds, x00, demand)

x_opt, LL0, LL_res, rho2, rho2_adj, tval_res = optimizer.optimize()
    
end_time = time.time()
proc_time = end_time - start_time

logger.info('---------- Estimation Results ----------')
logger.info('  process time = {}'.format(proc_time))
logger.info('timestep limit = {}'.format(timestep_limit))
logger.info('    init param = {}'.format(x00))
logger.info('   model param = {}'.format(x_opt))
logger.info('       t value = {}'.format(tval_res))
logger.info('           LL0 = {}'.format(LL0))
logger.info('            LL = {}'.format(LL_res))
logger.info('          rho2 = {}'.format(rho2))

