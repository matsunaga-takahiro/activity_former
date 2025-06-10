import torch
import pandas as pd
import numpy as np
from utils.logger import logger

from network import Network

torch.autograd.set_detect_anomaly(True)


class RecursiveLogit:
    def __init__(self, network, demand, discount_rate, T, device):
        self.network = network # ネットワーク
        self.demand = demand
        self.N_OD = len(demand)
        self.discount_rate = discount_rate  # 割引率
        self.T = T  # RLの時間ステップ数
        self.device = device  # 設備を指定 (CPU or GPU)

    def _Mset(self, x): # 時間ごと即時効用を計算する
        feature_mat_0 = torch.tensor(self.network.feature_mat_0, device=self.device, dtype=torch.float32)
        feature_mat_1 = torch.tensor(self.network.feature_mat_1, device=self.device, dtype=torch.float32)
        inst = feature_mat_0 * x[0] + feature_mat_1 * x[1]
        inst = inst.unsqueeze(0).repeat(self.T, 1, 1)
        return inst
    
    def newPall(self, x):
        '''
        選択確率の計算
        '''
        beta = self.discount_rate
        Pall = torch.zeros((self.N_OD, self.T, self.network.N, self.network.N), device=self.device)

        Mts = self._Mset(x)
        #print(Mts[0])
        M = torch.exp(Mts).to(self.device)
        #隣接しているもの以外の即時効用を0にする
        adj_matrix_expanded = self.network.adj_matrix.unsqueeze(0)
        adj_matrix_expanded = adj_matrix_expanded.to(self.device)
        M = M * adj_matrix_expanded

        for od in range(self.N_OD):
            d = self.demand[od, 1]
            z = torch.zeros((self.T + 1, self.network.N), device=self.device)  # 状態数は遷移数Tから1多いのでT+1
            #最後の状態ではdのみが1で他が0
            z[self.T, d] = 1

            d_mat = torch.ones((self.network.N, self.network.N), device=self.device)
            d_mat[d, :] = 0
            d_mat[d, d] = 1
            #目的地dからはdにしかいけないようにする
            M_d = M * d_mat
            
            for t in range(self.T, 0, -1):
                z_t_clamped = torch.clamp(z[t], min=1e-10)
                zii = M_d[t - 1] * (z_t_clamped.view(1, -1) ** beta)
                zi = zii.sum(axis=1)
                #z[t - 1] =zi
                z = torch.cat([z[:t - 1], zi.unsqueeze(0), z[t:]], dim=0)
                #z_new = z.clone()
                #zii = M_d[t - 1] * (z[t].view(1, -1) ** beta)
                #zi = zii.sum(axis=1)
                #z_new[t - 1] = zi


                #z[t - 1] = (zi == 0) * 1 + (zi != 0) * zi

            for t in range(self.T):
                M_d_clamped = torch.clamp(M_d[t], min=1e-10)
                z_t_plus_1_clamped = torch.clamp(z[t + 1], min=1e-10)
                z_t_clamped = torch.clamp(z[t], min=1e-10)
                transition_prob = (M_d_clamped * z_t_plus_1_clamped.view(1, -1).pow(beta)) / z_t_clamped.view(-1, 1)
                #Pall[od, t] = transition_prob
                Pall[od, t] = torch.where(transition_prob >= 1, torch.ones_like(transition_prob), transition_prob)

                #if (transition_prob > 1).any():
                    #logger.error(f'od={od}, t={t}')
                    #logger.error(f'M={M[t]}, z+={z[t + 1]}, z- = {z[t]}')
                    #logger.error(f'transition_prob={transition_prob}')
                    #logger.error(f'transition_prob={transition_prob}')
                #if (transition_prob < 0).any():
                    #logger.error(f'od={od}, t={t}')
                    #logger.error(f'M={M[t]}, z+={z[t + 1]}, z- = {z[t]}')
                    #logger.error(f'transition_prob={transition_prob}')
        return Pall

    def simulation(self, x):
        Pall = self.newPall(x)
        df = pd.DataFrame(columns = [str(i) for i in range(self.T + 1)] + ['od_pair'])
        for od in range(self.N_OD):
            o = self.demand[od, 0]
            simulation_num = self.demand[od, 2]
            simulation_result_tensor = torch.zeros((simulation_num, self.T + 1), device=self.device)
            for i in range(simulation_num):
                current_node = o
                simulation_result_tensor[i, 0] = current_node
                for t in range(self.T):
                    prob = Pall[od, t, current_node, :]
                    if prob.sum() <= 0:
                        logger.error(f'od={od}, t={t}, current_node={current_node}')
                        logger.error(f'Pall={Pall[od, t]}')
                    next_node = torch.multinomial(prob, 1).item()
                    simulation_result_tensor[i, t+1] = next_node
                    current_node = next_node
            result_df = pd.DataFrame(simulation_result_tensor.cpu().numpy(), columns = [str(i) for i in range(self.T + 1)])
            result_df['od_pair'] = od
            df = pd.concat([df, result_df])
        return df
    
class TimeFreeRecursiveLogit:
    def __init__(self, network, demand, discount_rate, T, device):
        self.network = network # ネットワーク
        self.demand = demand
        self.N_OD = len(demand)
        self.discount_rate = discount_rate  # 割引率
        self.T = T  # RLの時間ステップ数
        self.device = device  # 設備を指定 (CPU or GPU)

    def _Mset(self, x): # 時間ごと即時効用を計算する
        feature_mat_0 = torch.tensor(self.network.feature_mat_0, device=self.device, dtype=torch.float32)
        feature_mat_1 = torch.tensor(self.network.feature_mat_1, device=self.device, dtype=torch.float32)
        inst = feature_mat_0 * x[0] + feature_mat_1 * x[1]
        #吸収ノードのために列を追加する
        #回遊から抜けることに効用を持たせる
        new_col = torch.ones(self.network.N, 1, device=self.device)
        new_col *= 20
        inst = torch.cat([inst, new_col], dim=1) #zerosにした場合は効用なし，パラメータにして設定しても良い．
        inst = torch.cat([inst, torch.zeros((1, self.network.N + 1), device=self.device)], dim=0)
        #print(inst)

        inst = inst.unsqueeze(0).repeat(self.T, 1, 1)
        return inst
    
    def newPall(self, x):
        '''
        選択確率の計算
        '''
        beta = self.discount_rate
        N = self.network.N
        Pall = torch.zeros((self.N_OD, self.T, N + 1, N + 1), device=self.device)

        Mts = self._Mset(x)
        #print(Mts[0])
        M = torch.exp(Mts).to(self.device)
        #print(M[0])

        for od in range(self.N_OD):
            d = self.demand[od, 1]
            #隣接しているもの以外の即時効用を0にする
            #吸収ノードを追加
            adj_matrix = self.network.adj_matrix.to(self.device)
            new_row = torch.zeros(1, N, device=self.device)
            adj_matrix_expanded = torch.cat([adj_matrix, new_row], dim = 0)
            new_col = torch.zeros(N + 1, 1, device=self.device)
            adj_matrix_expanded = torch.cat([adj_matrix_expanded, new_col], dim = 1)
            adj_matrix_expanded[d, N] = 1
            adj_matrix_expanded[N, N] = 1
            adj_matrix_expanded = adj_matrix_expanded.unsqueeze(0)
            adj_matrix_expanded = adj_matrix_expanded.to(self.device)
            M = M * adj_matrix_expanded

            z = torch.zeros((self.T + 1, N+1), device=self.device)  # 状態数は遷移数Tから1多いのでT+1
            #最後の状態では吸収ノードのみが1で他が0
            z[self.T, N] = 1

            d_mat = torch.ones((N+1, N+1), device=self.device)
            d_mat[d, :] = 0
            d_mat[d, d] = 1
            d_mat[d, N] = 1
            #目的地dからはdにしかいけないようにする
            M_d = M * d_mat
            
            for t in range(self.T, 0, -1):
                z_t_clamped = torch.clamp(z[t], min=1e-10)
                zii = M_d[t - 1] * (z_t_clamped.view(1, -1) ** beta)
                zi = zii.sum(axis=1)
                #z[t - 1] =zi
                z = torch.cat([z[:t - 1], zi.unsqueeze(0), z[t:]], dim=0)
                #z_new = z.clone()
                #zii = M_d[t - 1] * (z[t].view(1, -1) ** beta)
                #zi = zii.sum(axis=1)
                #z_new[t - 1] = zi


                #z[t - 1] = (zi == 0) * 1 + (zi != 0) * zi

            for t in range(self.T):
                M_d_clamped = torch.clamp(M_d[t], min=1e-10)
                z_t_plus_1_clamped = torch.clamp(z[t + 1], min=1e-10)
                z_t_clamped = torch.clamp(z[t], min=1e-10)
                transition_prob = (M_d_clamped * z_t_plus_1_clamped.view(1, -1).pow(beta)) / z_t_clamped.view(-1, 1)
                #Pall[od, t] = transition_prob
                Pall[od, t] = torch.where(transition_prob >= 1, torch.ones_like(transition_prob), transition_prob)

                #if (transition_prob > 1).any():
                    #logger.error(f'od={od}, t={t}')
                    #logger.error(f'M={M[t]}, z+={z[t + 1]}, z- = {z[t]}')
                    #logger.error(f'transition_prob={transition_prob}')
                    #logger.error(f'transition_prob={transition_prob}')
                #if (transition_prob < 0).any():
                    #logger.error(f'od={od}, t={t}')
                    #logger.error(f'M={M[t]}, z+={z[t + 1]}, z- = {z[t]}')
                    #logger.error(f'transition_prob={transition_prob}')
        return Pall

    def simulation(self, x):
        Pall = self.newPall(x)
        df = pd.DataFrame(columns = [str(i) for i in range(self.T + 1)] + ['od_pair'])
        for od in range(self.N_OD):
            o = self.demand[od, 0]
            simulation_num = self.demand[od, 2]
            simulation_result_tensor = torch.zeros((simulation_num, self.T + 1), device=self.device)
            for i in range(simulation_num):
                current_node = o
                simulation_result_tensor[i, 0] = current_node
                for t in range(self.T):
                    prob = Pall[od, t, current_node, :]
                    if prob.sum() <= 0:
                        logger.error(f'od={od}, t={t}, current_node={current_node}')
                        logger.error(f'Pall={Pall[od, t]}')
                    next_node = torch.multinomial(prob, 1).item()
                    simulation_result_tensor[i, t+1] = next_node
                    current_node = next_node
            result_df = pd.DataFrame(simulation_result_tensor.cpu().numpy(), columns = [str(i) for i in range(self.T + 1)])
            result_df['od_pair'] = od
            df = pd.concat([df, result_df])
        return df