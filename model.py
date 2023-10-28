import torch
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv, global_mean_pool, GATv2Conv, GraphNorm
from torch_geometric.data import Data, Batch


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
        self, x_node_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor
    ) -> Tensor:
        # split and for loop to handle big number of graphs
        # node level features
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


class LinearActNorm(nn.Module):
    def __init__(self, input_dim, output_dim, act):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_normal_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)
        self.act = act(inplace=True)
        self.norm = GraphNorm(output_dim)
        self.attn = ChannelAttention(output_dim)

    def forward(self, x, edge_index, batch):
        # x = self.norm(self.act(self.linear(x)), batch)
        x = self.norm(self.linear(x), batch)
        x = self.attn(x)
        x = self.act(x)
        return x


class GATActBN(nn.Module):
    def __init__(self, channels, act, droprate=0.2, graph_droprate=0.1, nb_heads=2):
        super().__init__()
        # self.conv = GATv2Conv(
        #     channels, channels, heads=nb_heads, dropout=graph_droprate, concat=False
        # )
        self.conv = SAGEConv(channels, channels)
        self.act = act(inplace=True)
        self.bn = GraphNorm(channels)
        if droprate > 0:
            self.dropout = nn.Dropout(p=droprate)
        print(f"Inner Dropout {droprate}")
        self.droprate = droprate
        self.attn = ChannelAttention(channels)

    def forward(self, x, edge_index, batch):
        tmp = self.bn(self.conv(x, edge_index), batch)
        # tmp = self.bn(self.act(self.conv(x, edge_index)), batch)
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
        nb_heads=1,
        gat_droprate=0,
        graph_droprate=0,
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
        print(f"Dropout {dropout}")
        NODE_FEAT_DIM = 140
        NODE_CONFIG_FEAT_DIM = 18
        self.linear = LinearActNorm(
            op_embedding_dim + NODE_FEAT_DIM + (NODE_CONFIG_FEAT_DIM * layout_embedding_dim),
            graph_in,
            act=act,
        )

        self.convs = nn.ModuleList(
            [
                GATActBN(
                    graph_in,
                    act,
                    droprate=gat_droprate,
                    graph_droprate=graph_droprate,
                    nb_heads=nb_heads,
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

    def forward(
        self, x_node_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor
    ) -> Tensor:
        # split and for loop to handle big number of graphs
        # node level features
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
        # batch = Batch.from_data_list(
        #     [Data(x=x[i], edge_index=edge_index) for i in range(x.shape[0])]
        # )
        batch = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)

        x = self.linear(x, edge_index, batch=batch)  # .relu()

        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index, batch=batch)

        # put into dense nn
        # x = torch.flatten(self.dense(x))
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-5)

        # be careful that we have a batch of graphs with the same config here
        # x = global_mean_pool(x, batch.batch)
        x = x.mean(1)  # nb_configs, channels
        x = self.dense(x)
        # Cross config pooling
        x = x - x.mean(0)
        x = self.classifier(x)

        return x.reshape(-1)
