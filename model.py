import torch
from torch import Tensor, nn
import numpy as np
from torch_geometric.nn import (
    SAGEConv,
    global_mean_pool,
    GATv2Conv,
    GraphNorm,
    CuGraphSAGEConv,
    GENConv,
)
from torch_geometric.data import Data, Batch
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.utils import scatter
from collections import defaultdict


class FixedGraphNorm(torch.nn.Module):
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.empty(in_channels))
        self.bias = torch.nn.Parameter(torch.empty(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.empty(in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)

    def forward(self, x: Tensor, batch=None) -> Tensor:
        """X is of shape (nb_configs, nb_nodes, nb_features)"""
        mean = x.mean(dim=1, keepdim=True)
        out = (x - mean) * self.mean_scale
        var = out.pow(2)
        std = (var + self.eps).sqrt().mean(dim=1, keepdim=True)
        out = self.weight * out / std + self.bias
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels})"


class TileModel(torch.nn.Module):
    def __init__(self, hidden_channels, graph_in, graph_out, hidden_dim, dropout=0.0):
        super().__init__()
        op_embedding_dim = 4  # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(
            120,  # 120 different op-codes
            op_embedding_dim,
        )
        assert len(hidden_channels) > 0

        self.linear = nn.Linear(op_embedding_dim + 140, graph_in)
        in_channels = graph_in
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]
        conv = SAGEConv
        self.convs.append(conv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(conv(hidden_channels[i], hidden_channels[i + 1]))
            last_dim = hidden_channels[i + 1]
        self.convs.append(conv(last_dim, graph_out))

        self.dense = torch.nn.Sequential(
            nn.Linear(graph_out * 2 + 24, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    #         self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor) -> Tensor:
        # get graph features
        x = torch.concat([x_feat, self.embedding(x_op)], dim=1)
        x = self.linear(x)
        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        # get 1d graph embedding using average pooling
        x_mean = x.mean(0)
        x_max = x.max(0).values

        # put graph data into config data
        x = torch.concat(
            [x_cfg, x_max.repeat((len(x_cfg), 1)), x_mean.repeat((len(x_cfg), 1))],
            axis=1,
        )
        # put into dense nn
        x = torch.flatten(self.dense(x))
        x = (x - torch.mean(x)) / (torch.std(x) + 1e-5)
        return x


class LayoutModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        graph_in,
        graph_out,
        hidden_dim,
        dropout=0.0,
        op_embedding_dim=4,
        layout_embedding_dim=4,
    ):
        super().__init__()
        self.embedding_op = torch.nn.Embedding(
            120,  # 120 different op-codes
            op_embedding_dim,
        )
        self.embedding_layout = torch.nn.Embedding(
            5 + 3, layout_embedding_dim
        )  # [1-5] + [0,-1,-2]
        assert len(hidden_channels) > 0

        NODE_FEAT_DIM = 140
        NODE_CONFIG_FEAT_DIM = 18
        self.linear = nn.Linear(
            op_embedding_dim + NODE_FEAT_DIM + (NODE_CONFIG_FEAT_DIM * layout_embedding_dim),
            graph_in,
        )
        in_channels = graph_in
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]
        conv = SAGEConv
        self.convs.append(conv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(conv(hidden_channels[i], hidden_channels[i + 1]))
            last_dim = hidden_channels[i + 1]
        self.convs.append(conv(last_dim, graph_out))

        self.dense = torch.nn.Sequential(
            nn.Linear(graph_out, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_node_cfg: Tensor,
        x_feat: Tensor,
        x_op: Tensor,
        edge_index: Tensor,
        node_config_ids: Tensor,
    ) -> Tensor:
        # split and for loop to handle big number of graphs
        # node level features
        node_config_feat = (
            torch.ones(
                (x_node_cfg.shape[0], x_feat.shape[0], 18),
                dtype=torch.long,
                device=x_node_cfg.device,
            )
            * -2
        )
        node_config_feat[:, node_config_ids] = x_node_cfg
        node_config_feat = (
            node_config_feat + 2
        )  # -2 is min and 5 is max so offset to [0, 7] for embd layer

        # node_config_feat = node_config_feat / 3.0
        x_node_cfg = node_config_feat

        # x_node_cfg (num_configs, num_nodes, 18)
        # x_feat (num_nodes, 140)
        # x_op (num_nodes,)

        x_node_cfg = self.embedding_layout(x_node_cfg)  # (num_configs, num_nodes, 18, embd_width)
        x_node_cfg = x_node_cfg.view(
            x_node_cfg.shape[0], x_node_cfg.shape[1], -1
        )  # (num_configs, num_nodes, 18*embd_width)
        x_feat = x_feat.unsqueeze(0).repeat(
            (x_node_cfg.shape[0], 1, 1)
        )  # (num_configs, num_nodes, 140)
        x_op = (
            self.embedding_op(x_op).unsqueeze(0).repeat((x_node_cfg.shape[0], 1, 1))
        )  # (num_configs, num_nodes, embd_width)

        node_feat = torch.concat([x_feat, x_node_cfg], dim=2)
        x = torch.concat(
            [node_feat, x_op],
            dim=2,
        )  # (num_configs, num_nodes, 140+(18*embd_layout_width)+embd_op_width)
        x = self.linear(x)  # .relu()

        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()  # + x

        # put into dense nn
        # x = torch.flatten(self.dense(x))
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-5)

        # be careful that we have a batch of graphs with the same config here
        x = x.mean(1)
        x = self.dense(x)

        return x.reshape(-1)


class ChannelAttention(nn.Module):
    def __init__(self, input_dim, act=nn.ReLU, reduction=8):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, int(input_dim / reduction))
        self.linear2 = nn.Linear(int(input_dim / reduction), input_dim)
        self.sigmoid = nn.Sigmoid()
        self.act = act(inplace=True)

    def forward(self, x):
        tmp = self.act(self.linear1(x))
        tmp = self.linear2(tmp)
        tmp = self.sigmoid(tmp)
        return tmp * x


class InstanceNorm1d(nn.InstanceNorm1d):
    def forward(self, x, batch):
        x = x.swapaxes(1, 2)
        x = super().forward(x)
        return x.swapaxes(1, 2)


class LinearActNorm(nn.Module):
    def __init__(self, input_dim, output_dim, act, norm="instance"):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)
        self.act = act(inplace=True)
        if norm == "instance":
            self.norm = InstanceNorm1d(output_dim)
        else:
            self.norm = GraphNorm(output_dim)
        self.attn = ChannelAttention(output_dim)

    def forward(self, x, edge_index, batch):
        x = self.norm(self.linear(x), batch)
        # x = self.linear(x)
        x = self.attn(x)
        x = self.act(x)
        return x


class SageGraphBlock(nn.Module):
    def __init__(
        self, channels, act, droprate=0.2, graph_droprate=0.1, nb_heads=2, norm="instance"
    ):
        super().__init__()
        # self.conv = GATv2Conv(
        #     channels, channels, heads=nb_heads, dropout=graph_droprate, concat=False
        # )
        self.conv = SAGEConv(channels, channels)
        self.act = act(inplace=True)
        if norm == "instance":
            self.norm = InstanceNorm1d(channels)
        else:
            self.norm = GraphNorm(channels)
        if droprate > 0:
            self.dropout = nn.Dropout(p=droprate)
        print(f"Inner Dropout {droprate}")
        self.droprate = droprate
        self.attn = ChannelAttention(channels)

    def forward(self, x, edge_index, batch):
        tmp = self.norm(self.conv(x, edge_index), batch)
        # tmp = self.conv(x, edge_index)
        x = self.attn(x)
        tmp = tmp + x
        tmp = self.act(tmp)
        if self.droprate > 0:
            tmp = self.dropout(tmp)
        return tmp


class GATLayoutModel(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        graph_in,
        graph_out,
        hidden_dim,
        dropout=0.0,
        op_embedding_dim=4,
        layout_embedding_dim=4,
        act=nn.ReLU,
        norm="instance",
        nb_heads=1,
        gat_droprate=0,
        graph_droprate=0,
    ):
        super().__init__()
        self.embedding_op = torch.nn.Embedding(
            120,  # 120 different op-codes
            op_embedding_dim,
        )
        self.embedding_layout_cfg = torch.nn.Embedding(
            5 + 2, layout_embedding_dim
        )  # [1-5] + [0,-1]
        # self.embedding_layout_feats = torch.nn.Embedding(6, layout_embedding_dim)  # [0-5]
        assert len(hidden_channels) > 0
        print(f"Dropout {dropout}")
        NODE_FEAT_DIM = 134
        NODE_CONFIG_FEAT_DIM = 18
        NODE_LAYOUT_FEAT_DIM = 6
        self.linear = LinearActNorm(
            op_embedding_dim
            + NODE_FEAT_DIM
            + (NODE_LAYOUT_FEAT_DIM * layout_embedding_dim)
            + (NODE_CONFIG_FEAT_DIM * layout_embedding_dim),
            graph_in,
            act=act,
            norm=norm,
        )

        self.convs = nn.ModuleList(
            [
                SageGraphBlock(
                    graph_in,
                    act,
                    droprate=gat_droprate,
                    graph_droprate=graph_droprate,
                    nb_heads=nb_heads,
                    norm=norm,
                )
                for _ in range(len(hidden_channels))
            ]
        )
        self.dense = torch.nn.Sequential(
            nn.Linear(graph_out, hidden_dim),
            ChannelAttention(hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            ChannelAttention(hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def single_pass_forward(
        self,
        x_node_cfg_j: Tensor,
        x_feat: Tensor,
        x_node_layout_feat: Tensor,
        x_op: Tensor,
        edge_index: Tensor,
        node_config_ids: Tensor,
    ):
        node_cfg_feat_j = (
            torch.ones(
                (x_node_cfg_j.shape[0], x_feat.shape[0], 18),
                dtype=torch.long,
                device=x_node_cfg_j.device,
            )
            * -1
        )
        node_cfg_feat_j[:, node_config_ids] = x_node_cfg_j
        node_cfg_feat_j = (
            node_cfg_feat_j + 1
        )  # -1 is min and 5 is max so offset to [0, 6] for embd layer

        node_cfg_feat_j = self.embedding_layout_cfg(
            node_cfg_feat_j
        )  # (num_configs, num_nodes, 18, embd_width)
        node_cfg_feat_j = node_cfg_feat_j.view(
            node_cfg_feat_j.shape[0], node_cfg_feat_j.shape[1], -1
        )  # (num_configs, num_nodes, 18*embd_width)
        x_feat_j = x_feat.unsqueeze(0).repeat(
            (node_cfg_feat_j.shape[0], 1, 1)
        )  # (num_configs, num_nodes, 140)
        x_op_j = (
            self.embedding_op(x_op).unsqueeze(0).repeat((node_cfg_feat_j.shape[0], 1, 1))
        )  # (num_configs, num_nodes, embd_width)

        # TODO: Right now `node_layout_feat_embd` has almost no value since it was 0 padded instead
        # of -1 padded like `node_cfg_feat_j`. Once we fix the data extraction we can probably use
        # this. I think this will be an important feature since the combination of this + input/output
        # config feats is what dictates if the copy operation are added or not.

        node_layout_feat_embd = self.embedding_layout_cfg(x_node_layout_feat + 1)
        node_layout_feat_embd = (
            node_layout_feat_embd.unsqueeze(0)
            .view(1, node_cfg_feat_j.shape[1], -1)
            .repeat((node_cfg_feat_j.shape[0], 1, 1))
        )

        node_feat = torch.concat(
            [x_feat_j, node_cfg_feat_j, node_layout_feat_embd, x_op_j], dim=2
        )  # (num_configs, num_nodes, 140+(18*embd_layout_width)+embd_op_width)

        # Only used for GraphNorm, which takes stats across the nb_configs dimension
        batch = torch.zeros(node_feat.shape[0], dtype=torch.long).to(node_feat.device)

        node_feat = self.linear(node_feat, edge_index, batch)  # .relu()

        # pass though conv layers
        for conv in self.convs:
            node_feat = conv(node_feat, edge_index, batch)

        node_feat = node_feat.mean(1)  # nb_configs, channels
        node_feat = self.dense(node_feat)
        return node_feat

    def forward(
        self,
        x_node_cfg: Tensor,
        x_feat: Tensor,
        x_node_layout_feat: Tensor,
        x_op: Tensor,
        edge_index: Tensor,
        node_config_ids: Tensor,
    ) -> Tensor:
        # split and for loop to handle big number of graphs
        # node level features
        # x_node_cfg (num_configs, num_nodes, 18)
        # x_feat (num_nodes, 140)
        # x_op (num_nodes,)
        # if self.training:
        embedding = self.single_pass_forward(
            x_node_cfg, x_feat, x_node_layout_feat, x_op, edge_index, node_config_ids
        )
        return self.classifier(embedding).reshape(-1)

        # # Inference time first pass
        # nb_configs = x_node_cfg.shape[0]
        # nb_iters = int(np.ceil(nb_configs / self.max_configs))
        # embeddings = []
        # for j in range(nb_iters):
        #     x_node_cfg_j = x_node_cfg[j * self.max_configs : (j + 1) * self.max_configs]
        #     embedding = self.single_pass_forward(
        #         x_node_cfg_j,
        #         x_feat,
        #         x_op,
        #         edge_index,
        #         node_config_ids,
        #     )
        #     embeddings.append(embedding)
        # embeddings = torch.cat(embeddings, dim=0)
        # # Cross config pooling
        # logits = self.classifier(embeddings)
        # return logits.reshape(-1)
