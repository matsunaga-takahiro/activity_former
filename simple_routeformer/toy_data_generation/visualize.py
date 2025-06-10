import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import torch
from collections import Counter

adj_matrix = torch.load("grid_adjacency_matrix.pt")
node_features = torch.load("node_features_matrix.pt")
route_df = pd.read_csv("route_data/route.csv", index_col=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestep_limit = 10

# グラフの作成
G = nx.Graph()  # 有向グラフの場合は nx.DiGraph() を使用

# 隣接行列からエッジを追加
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if adj_matrix[i, j] > 0:  # エッジが存在する場合
            G.add_edge(i, j)

# oとdを指定する
o = 0
d = 15
route_df = route_df[(route_df["0"] == o) & (route_df[str(timestep_limit)] == d)]
paths = route_df.loc[:, "1":"9"].values.tolist()

# ノード通過回数をカウント
node_visits = Counter()
for path in paths:
    node_visits.update(path)

# カウント結果を確認
print(node_visits)

# ノードごとの通過回数をリスト化
node_colors = [node_visits.get(node, 0) for node in G.nodes]
node_sizes = [node_visits.get(node, 0) * 100 for node in G.nodes]  # サイズ調整

fixed_node_size = 500  # 固定ノードサイズ


# グラフ描画
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
nodes = nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=fixed_node_size,
    cmap=plt.cm.Blues
)

# カラーバーの作成
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
sm.set_array(node_colors)  # ノードデータをセット
plt.gca().set_title("Node Visit Visualization")
cbar = plt.colorbar(sm, ax=plt.gca(), label="Node Visits")  # カラーバーに現在のAxesを明示的に指定


# タイトルと保存
plt.title("Node Visit Visualization")
plt.savefig("visualize/node_visits.png")

