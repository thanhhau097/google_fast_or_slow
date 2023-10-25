import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv, global_mean_pool


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

    def forward(
        self, x_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor
    ) -> Tensor:
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
        max_configs=128,
    ):
        super().__init__()
        # # per feature layers
        # general_embedding_dim = 8
        # self.shape_dimensions_layer = nn.Linear(19, general_embedding_dim)

        op_embedding_dim = 4  # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(
            120,  # 120 different op-codes
            op_embedding_dim,
        )
        assert len(hidden_channels) > 0

        NODE_FEAT_DIM = 140
        NODE_CONFIG_FEAT_DIM = 18
        self.linear = nn.Linear(
            op_embedding_dim + NODE_FEAT_DIM + NODE_CONFIG_FEAT_DIM, graph_in
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
                dtype=torch.float32,
                device=x_node_cfg.device,
            )
            * -2
        )
        node_config_feat[:, node_config_ids] = x_node_cfg
        x_node_cfg = node_config_feat

        node_feat = torch.concat(
            [x_feat.unsqueeze(0).repeat((x_node_cfg.shape[0], 1, 1)), x_node_cfg], dim=2
        )
        x = torch.concat(
            [
                node_feat,
                self.embedding(x_op).unsqueeze(0).repeat((x_node_cfg.shape[0], 1, 1)),
            ],
            dim=2,
        )
        x = self.linear(x)

        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # put into dense nn
        # x = torch.flatten(self.dense(x))
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-5)

        # be careful that we have a batch of graphs with the same config here
        x = x.mean(1)
        x = self.dense(x)

        return x.reshape(-1)
