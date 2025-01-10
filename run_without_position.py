import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        if in_features != out_features:
            self.input_proj = nn.Linear(in_features, out_features)
        else:
            self.input_proj = nn.Identity()

        self.query = nn.Linear(out_features, out_features)
        self.key = nn.Linear(out_features, out_features)
        self.value = nn.Linear(out_features, out_features)

        self.mha_proj = nn.Linear(out_features, out_features)

        self.ffn = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.ReLU(),
            nn.Linear(out_features * 2, out_features),
        )

        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

    def forward(self, x, edge_index):
        x = self.input_proj(x)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        attn_scores = torch.einsum("nhd,mhd->nhm", q, k) / (self.head_dim ** 0.5)
        adj_matrix = torch.zeros(x.size(0), x.size(0), device=x.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        attn_scores = attn_scores.masked_fill(adj_matrix.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=1)
        attn_output = torch.einsum("nhm,mhd->nhd", attn_weights, v)
        attn_output = attn_output.reshape(-1, self.num_heads * self.head_dim)
        attn_output = self.mha_proj(attn_output)

        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)
        return x


class GraphTransformer(nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features, num_heads):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(in_features if i == 0 else hidden_features, hidden_features, num_heads)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def run_train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def run_test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    return acc


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU，设置所有 GPU 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    random.seed(seed)  # 设置 Python 内置随机模块的随机种子


def run():
    # dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"data: {data.x}")

    num_nonzero = torch.count_nonzero(data.x)
    num_total = data.x.numel()
    sparsity = 1 - (num_nonzero / num_total)
    print(f"Number of non-zero elements: {num_nonzero}")
    print(f"Total number of elements: {num_total}")
    print(f"Sparsity: {sparsity:.4f}")

    # 设置随机种子
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = GraphTransformer(
        num_layers=1,
        in_features=dataset.num_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_heads=4
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)

    best_acc = -1
    no_best_count = 0
    for epoch in range(100000):
        loss = run_train(model, data, optimizer, criterion)
        if epoch % 10 == 0:
            acc = run_test(model, data)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')
            if acc > best_acc:
                best_acc = acc
                no_best_count = 0
            else:
                no_best_count += 1
                if no_best_count >= 100:
                    break
    print(f'Best Accuracy: {best_acc:.4f}')



if __name__ == '__main__':
    run()
