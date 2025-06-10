# generation of simulation data of activity scheduling

import os
import pandas as pd
import numpy as np
from setting import *
from recursive import *


base_path = '/Users/matsunagatakahiro/Desktop/res2025/ActFormer/toyact_gen'

def main():
    print("***********generation start***********")

    df_node, df_indivi = readfile(base_path)
    true_param = np.array([-1, 0.5, 1.5, -0.5, 0.5, 2, 1]) 
    network = Network(df_indivi, df_node, T = 19) # 24-6+1 = 19
    recursive = Recursive(network, df_route = None)
    df_route_assigned, df_state_assigned, df_act_assigned = recursive.assign(true_param)
    df_route_assigned.to_csv(os.path.join(base_path, 'output/route_traj.csv'), index = False)
    df_state_assigned.to_csv(os.path.join(base_path, 'output/state_traj.csv'), index = False)
    df_act_assigned.to_csv(os.path.join(base_path, 'output/act_traj.csv'), index = False)
    # /Users/matsunagatakahiro/Desktop/res2025/ActFormer/toyact_gen/main_gen.py
    print("***********generation end***********")

def readfile(base_path):
    indivi_path = os.path.join(base_path, 'input/indivi.csv')
    node_path = os.path.join(base_path, 'input/node.csv')

    df1 = pd.read_csv(node_path)
    df2 = pd.read_csv(indivi_path)

    return df1, df2

    
if __name__ == '__main__':
    main()