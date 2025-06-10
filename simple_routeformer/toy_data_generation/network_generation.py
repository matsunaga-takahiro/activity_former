import torch

def create_grid_adjacency_matrix_torch(N):
    """
    Create the adjacency matrix for an NxN grid network using PyTorch tensors.

    Parameters:
        N (int): The size of the grid (N x N).

    Returns:
        torch.Tensor: The adjacency matrix of the grid.
    """
    # Number of nodes
    num_nodes = N * N

    # Initialize the adjacency matrix with zeros
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.int)

    # Helper function to convert (row, col) to node index
    def node_index(row, col):
        return row * N + col

    # Fill the adjacency matrix
    for row in range(N):
        for col in range(N):
            dia = row * N + col
            adjacency_matrix[dia, dia] = 1
            current = node_index(row, col)
            # Connect to the node above
            if row > 0:
                above = node_index(row - 1, col)
                adjacency_matrix[current, above] = 1
                adjacency_matrix[above, current] = 1
            # Connect to the node below
            if row < N - 1:
                below = node_index(row + 1, col)
                adjacency_matrix[current, below] = 1
                adjacency_matrix[below, current] = 1
            # Connect to the node to the left
            if col > 0:
                left = node_index(row, col - 1)
                adjacency_matrix[current, left] = 1
                adjacency_matrix[left, current] = 1
            # Connect to the node to the right
            if col < N - 1:
                right = node_index(row, col + 1)
                adjacency_matrix[current, right] = 1
                adjacency_matrix[right, current] = 1

    return adjacency_matrix


#ノードごとの特徴ベクトル

def generate_node_features(num_nodes, feature_dim):

    # 0から1の乱数を生成
    node_features = torch.rand(num_nodes, feature_dim)
    return node_features

# Example usage
N = 4  # Grid size (4x4 grid)
feature_dim = 1
num_nodes = N * N
adj_matrix = create_grid_adjacency_matrix_torch(N)
print("Adjacency Matrix for a {}x{} Grid:".format(N, N))
print(adj_matrix)
torch.save(adj_matrix, '/home/kurasawa/master_code/transformer/data_generation/grid_adjacency_matrix.pt')

#ノードの特徴行列を作る
# 連続量の特徴ベクトル
cont_vec1 = torch.tensor([0.5, 0.4, 0.2, 0, 0.6, 0.5, 0.4, 0.2, 0.8, 0.6, 0.5, 0.4, 1, 0.8, 0.6, 0.5])
# ダミーの特徴ベクトル
dummy_vec = torch.tensor([1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])

node_features = torch.cat((cont_vec1.view(-1,1), dummy_vec.view(-1,1)), dim=1)

# 結果を表示
print("Generated Node Features:")
print(node_features)
torch.save(node_features, '/home/kurasawa/master_code/transformer/data_generation/node_features_matrix.pt')