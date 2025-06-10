import numpy as np

    
class Network:
    def __init__(self, df_indivi, df_node, T):
        self.df_indivi = df_indivi
        self.df_node = df_node
        self.OD = len(self.df_indivi) # 個人の数
        self.T = T # 時間の数 # 24-6+1=19
        self.N = 4 # node数
        self.A = 4 # act数(home, shopping, work, leisure)
        self.SA = self.N * self.A # 状態数×活動数の同時分布

        self.I = None
        self.dist_mat = None
        self.shop_mat = None
        self.shop_mat_continu = None
        self.leisure_mat = None
        self.leisure_mat_continu = None
        self.home_mat = None
        self.home_mat_continu = None

    
    # 説明変数行列を用意する必要がある
    def prism(self): # 行動制約 # 説明変数行列の用意
        I = np.ones((self.OD, self.T, self.SA, self.SA))
        dist_mat_t = np.zeros((self.SA, self.SA))
        shop_mat_t = np.zeros((self.SA, self.SA))

        # home: 個人ごとに自宅ノードの自宅活動
        home_mat_t = np.zeros((self.OD, self.SA, self.SA))
        home_mat_continu_t = np.zeros((self.OD, self.SA, self.SA))
        
        for od in range(self.OD):
            if self.df_indivi.loc[od, 'worktime'] != 0: # worktimeが0ならバンドル制約なし
                workstart = self.df_indivi.loc[od, 'workstart']
                workend = self.df_indivi.loc[od, 'workend']
                worknode = self.df_indivi.loc[od, 'worknode']

                workidx = self.A * worknode + 1 # worknodeのworkのみが利用可能
                I[od, workstart-6:workend-6+1, :, :] = 0
                I[od, workstart-6:workend-6+1, workidx, workidx] = 1 # workstart-6:workend-6はworknodeのworkのみが利用可能 # バンドル制約
                I[od, workstart-6, :, workidx] = 1 # どこからでも出勤できる
                I[od, workend-6, workidx, :] = 1 # 退勤したらどこにでも行ける

            homenode = self.df_indivi.loc[od, 'homenode']
            homeidx = self.A * homenode
            I[od, 0, :, :] = 0
            I[od, -1, :, :] = 0
            I[od, 1, :, :] = 0
            I[od, -2, :, :] = 0
            I[od, 0, homeidx, homeidx] = 1 # 自宅ノードで自宅活動のみ1
            I[od, 1, homeidx, :] = 1
            I[od, -1, homeidx, homeidx] = 1 # 自宅ノードで自宅活動のみ1
            I[od, -2, :, homeidx] = 1

            home_mat_continu_t[od, homeidx, homeidx] = 1
            home_mat_t[od, :, homeidx] = 1 # 自宅ノードに行くところだけ1

        # home_mat = np.tile(home_mat_t, (self.T, 1)).reshape((self.T, self.OD, self.SA, self.SA))
        # home_mat_continu = np.tile(home_mat_continu_t, (self.T, 1)).reshape((self.T, self.OD, self.SA, self.SA))
        home_mat = np.tile(home_mat_t[:, np.newaxis, :, :], (1, self.T, 1, 1))  # (OD, T, SA, SA)
        home_mat_continu = np.tile(home_mat_continu_t[:, np.newaxis, :, :], (1, self.T, 1, 1))  # (OD, T, SA, SA)

        for i in range(self.N):
            for j in range(self.N):
                coord1 = (self.df_node.loc[i, 'x'], self.df_node.loc[i, 'y'])
                coord2 = (self.df_node.loc[j, 'x'], self.df_node.loc[j, 'y'])
                dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
                dist_mat_t[i*self.A : (i+1)*self.A, j*self.A : (j+1)*self.A] = dist

        # shopping: node3でのみ利用可能 継続と分ける
        shop_mat_t = np.zeros((self.SA, self.SA))
        shop_mat_t_continu = np.zeros((self.SA, self.SA))
        shopidx = self.A * 3 + 2 # node3の3番目のみで利用可能
        shop_mat_t[:, shopidx] = 1 # shoppingは3番目
        shop_mat_t[shopidx, shopidx] = 0
        shop_mat_t_continu[shopidx, shopidx] = 1 # shoppingの継続

        # leisure: node0の4番目のみで利用可能 継続と分ける
        leisureidx = self.A * 0 + 4 
        leisure_mat_t = np.zeros((self.SA, self.SA))
        leisure_mat_t_continu = np.zeros((self.SA, self.SA))
        leisure_mat_t[:, leisureidx] = 1
        leisure_mat_t[leisureidx, leisureidx] = 0
        leisure_mat_t_continu[leisureidx, leisureidx] = 1

        dist_mat = np.tile(dist_mat_t, (self.T, 1)).reshape((self.T, self.SA, self.SA))
        shop_mat = np.tile(shop_mat_t, (self.T, 1)).reshape((self.T, self.SA, self.SA))
        shop_mat_continu = np.tile(shop_mat_t_continu, (self.T, 1)).reshape((self.T, self.SA, self.SA))
        leisure_mat = np.tile(leisure_mat_t, (self.T, 1)).reshape((self.T, self.SA, self.SA))
        leisure_mat_continu = np.tile(leisure_mat_t_continu, (self.T, 1)).reshape((self.T, self.SA, self.SA))


        self.I = I
        self.dist_mat = dist_mat
        self.shop_mat = shop_mat
        self.shop_mat_continu = shop_mat_continu
        self.leisure_mat = leisure_mat
        self.leisure_mat_continu = leisure_mat_continu
        self.home_mat = home_mat
        self.home_mat_continu = home_mat_continu

        return I, dist_mat, shop_mat, shop_mat_continu, leisure_mat, leisure_mat_continu, home_mat, home_mat_continu
    
