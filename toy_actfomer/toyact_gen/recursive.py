import pandas as pd 
import numpy as np
import random


class Recursive:
    def __init__(self, network, df_route): # 推定の時df_demandは空，配分の時df_route は空
        self.network = network
        self.network.prism()
        self.df_route = df_route
        self.df_route_assigned = None
        self.df_state_assigned = None
        self.df_act_assigned = None


    def Mset(self, param): # ベースの即時効用（全員共通）
        x = param
        inst = np.zeros((self.network.T, self.network.SA, self.network.SA))
        # for t in range(self.network.T):
        inst = np.exp(self.network.dist_mat * x[0]
                        + self.network.shop_mat * x[1]
                        + self.network.leisure_mat * x[2]
                        + self.network.shop_mat_continu * x[3]
                        + self.network.leisure_mat_continu * x[4]
                        )
        return inst
    

    def newPall(self, param): # 個人ごとに定義する # 外の時間ttごとに選択確率が変わる設定
        x = param
        Pall = np.zeros((self.network.OD, self.network.T, self.network.SA, self.network.SA)) # absまでの遷移回数はT 個人時刻毎の遷移確率行列 遷移回数なのでT-1．今回Tを状態の数としているので遷移数はT-1になる
        beta = 0.9 
        M_base = self.Mset(x)

        for od in range(self.network.OD): # 個人ごとの考慮が入る
            Id = self.network.I[od, :, :, :] # 個人ごとの制約行列
            # pair_list = self.network.NPlist[od]
            M = np.zeros((self.network.T, self.network.SA, self.network.SA))
            for ts in range(self.network.T):
                Mts = Id[ts, :, :] * M_base[ts, :, :] * np.exp(self.network.home_mat[od, ts, :, :] * x[5] + self.network.home_mat_continu[od, ts, :, :] * x[6])
                M[ts, :, :] = Mts

            z = np.ones((self.network.T + 1, self.network.SA)) # 状態数は遷移数Tから1多いのでT+1
            for t in range(self.network.T, 0, -1):
                zii = M[t-1, :, :] * (z[t, :] ** beta) 
                zi = zii.sum(axis = 1)
                z[t-1, :] = (zi == 0) * 1 + (zi != 0) * zi

            for t in range(self.network.T): 
                for k in range(self.network.SA):
                    for a in range(self.network.SA):
                        if M[t, k, a] == 0:
                            continue
                        Pall[od, t, k, a] += np.exp(np.log(M[t, k, a]) + beta * np.log(z[t+1, a]) - np.log(z[t, k])) 
        return Pall 


    # def likelihood(self, param):
    #     x = param
    #     kt = 0
    #     kt = 0
    #     LL = 0
    #     for od in range(self.network.OD): # 個人ごとに属性で変わるはずなので
    #         for t in range(1, self.network.T):
    #             kt = int(self.df_route.loc[od * self.network.T + t - 1, 'k']) # nodeid
    #             at = int(self.df_route.loc[od * self.network.T + t, 'k'])
    #             Pall = self.newPall(x)
    #             p = Pall[od, t, kt-1, at-1]
    #             p = (p == 0) + (p != 0) * p
    #             LL += np.log(p)
    #     return -LL 


    def assign(self, param):
        x = param
        Pall = self.newPall(x)
        res_all = np.zeros(self.network.T) # userid, time, nodeid
        res_state_all = np.zeros(self.network.T) 
        res_act_all = np.zeros(self.network.T)

        # 個人ごと時間ごとに配分していく
        for od in range(self.network.OD):
            Pod = Pall[od, :, :, :] # 累積確率行列
            homenodeid = self.network.df_indivi.loc[od, 'homenode'] # 初期位置
            k = self.network.A * homenodeid # 初期ノード
            res_indivi = np.zeros(self.network.T) # 横に状態を並べるイメージ
            res_state_indivi = np.zeros(self.network.T)
            res_act_indivi = np.zeros(self.network.T)
            # res_indivi[:, 0] = od
            # res_indivi[:, 1] = np.arange(self.network.T)
            # res_indivi[0, 2] = k
            res_indivi[0] = k # 初期ノード
            res_state_indivi[0] = homenodeid
            res_act_indivi[0] = 0 # 自宅は0
            ran_list = [random.random() for _ in range(self.network.T)]

            # 累積確率行列
            for t in range(self.network.T):
                for sa in range(self.network.SA):
                    if sa == 0:
                        Pod[t, :, sa] = Pod[t, :, sa]
                    else:
                        Pod[t, :, sa] += Pod[t, :, sa-1]
            
            # 確率的配分
            for t in range(1, self.network.T):
                ran = ran_list[t]
                k = int(res_indivi[t-1])
                for a in range(self.network.SA):
                    if ran <= Pod[t, k, a]:
                        res_indivi[t] = a
                        res_state_indivi[t] = a // self.network.A
                        res_act_indivi[t] = a % self.network.A
                        break
            res_all = np.vstack((res_all, res_indivi))
            res_state_all = np.vstack((res_state_all, res_state_indivi))
            res_act_all = np.vstack((res_act_all, res_act_indivi))
        
        res_all = np.delete(res_all, 0, axis = 0)
        res_state_all = np.delete(res_state_all, 0, axis = 0)
        res_act_all = np.delete(res_act_all, 0, axis = 0)

        df_route_assigned = pd.DataFrame(res_all, columns=[t for t in range(self.network.T)]) # indexは個人ごとにする
        df_state_assigned = pd.DataFrame(res_state_all, columns=[t for t in range(self.network.T)]) # indexは個人ごとにする
        df_act_assigned = pd.DataFrame(res_act_all, columns=[t for t in range(self.network.T)])

        self.df_route_assigned = df_route_assigned
        self.df_state_assigned = df_state_assigned
        self.df_act_assigned = df_act_assigned

        return df_route_assigned, df_state_assigned, df_act_assigned
