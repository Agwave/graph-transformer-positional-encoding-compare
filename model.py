import torch
import torch.nn as nn
import torch.nn.functional as F


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
        attn_scores = attn_scores.masked_fill(adj_matrix.unsqueeze(1) == 0, float('-inf'))

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
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
