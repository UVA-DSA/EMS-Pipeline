from torch_geometric.nn import HGTConv, GATConv, Linear
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
from .default_sets import p_node, ungrouped_p_node, hier, device

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(-1, hidden_channels, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class HGT(nn.Module):
    def __init__(self, hetero, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in hetero.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, hetero.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, G):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in G.x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, G.edge_index_dict)

        graph_feat = {
            'protocols': self.lin(x_dict['protocol']),
            'impressions': self.lin(x_dict['impression']),
            'medications': self.lin(x_dict['medication']),
            'procedures': self.lin(x_dict['procedure'])
        }
        # return self.lin(x_dict['protocol'])
        return graph_feat

class EMSBERT(nn.Module):
    def __init__(self, bert_model):
        super(EMSBERT, self).__init__()
        self.bert = bert_model

    def forward(self, ids, mask):
        """
        BERT outputs:
        last_hidden_states: (b, t, h)
        pooled_output: (b, h), from output of a linear classifier + tanh
        hidden_states: 13 x (b, t, h), embed to last layer embedding
        attentions: 12 x (b, num_heads, t, t)
        """
        outputs = self.bert(ids, attention_mask=mask)
        return outputs[0], outputs[1]


class EMSMultiModel(nn.Module):
    def __init__(self, backbone, dp, max_len, attn, cluster, cls, graph):
        super(EMSMultiModel, self).__init__()
        self.bert = EMSBERT(BertModel.from_pretrained(backbone, output_hidden_states=True))
        self.hidden_size = 768
        self.fusion = None
        self.cluster = cluster
        if self.cluster == 'group':
            self.num_class = len(p_node) #[hier, p_node]
        elif self.cluster == 'ungroup':
            self.num_class = len(ungrouped_p_node)
        else:
            raise Exception('mode can only be [group, ungroup]')
        self.seq_length = max_len
        self.attn = attn
        self.cls = cls
        # self.graph = graph

        if graph:
            self.graph_model = HGT(graph, hidden_channels=256, out_channels=self.hidden_size, num_heads=8, num_layers=1)
            self.graph_model.to(device)
        else:
            self.graph_model = None

        if self.attn == 'qkv-la':
            self.attn_net = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=2, batch_first=True)
        elif self.attn == 'sa':
            self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)
            self.linear1 = nn.Linear(self.hidden_size, 256)
            self.linear2 = nn.Linear(4 * 256, self.hidden_size)
            self.tanh = nn.Tanh()
        elif self.attn == 'la':
            self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.tanh = nn.Tanh()

        if self.cls != None:
            if self.attn != None:
                input_dim = self.num_class * self.hidden_size
            else:
                input_dim = 768
            self.fc = nn.Linear(input_dim, self.num_class)
            # self.fc = nn.Sequential(
            #     nn.Linear(input_dim, 1024),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(p=dp),
            #     nn.Linear(1024, 256),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout(p=dp),
            #     nn.Linear(256, self.num_class)
            # )

    def _fusion(self, text_feat, graph_feat):
        # text_feat: batch_size(16) * class_num(52) * hidden_size(768) or batch_size(16) * hidden_size(768)
        # graph_feat: class_num(52) * 768
        # element-wise dot product -> sum
        out = None
        if self.fusion == 'element-wise product':
            b = text_feat.shape[0]
            graph_feat = graph_feat.unsqueeze(0).repeat(b, 1, 1)
            out = torch.mul(text_feat, graph_feat)  # (batch_size, class_num, hidden_size)
        # element-wise add -> sum
        elif self.fusion == 'element-wise add':
            b = text_feat.shape[0]
            graph_feat = graph_feat.unsqueeze(0).repeat(b, 1, 1)
            out = torch.add(text_feat, graph_feat)  # (batch_size, class_num, hidden_size)
        elif self.fusion == 'linear approximation':
            b = text_feat.shape[0]
            graph_feat = graph_feat.unsqueeze(0).repeat(b, 1, 1)  # (batch_size, class_num, hidden_size)
            gram = torch.bmm(graph_feat, graph_feat.permute(0, 2, 1))  # (batch_size, num_class, num_class)
            weights = torch.bmm(text_feat, graph_feat.permute(0, 2, 1))  # (batch_size, num_class, num_class)
            weights = torch.bmm(weights, torch.inverse(gram))  # (batch_size, num_class, num_class)
            out = torch.bmm(weights, graph_feat)  # (batch_size, num_class, hidden_size)
        elif self.fusion == 'concatenate':
            b = text_feat.shape[0]
            graph_feat = graph_feat.unsqueeze(0).repeat(b, 1, 1)
            out = torch.cat((text_feat, graph_feat), dim=-1)
        else:
            raise Exception('mode can only be [element-wise product, element-wise add]')
        return out

    def attn_network(self, last_hidden_state, label_feat):
        attn_output = None
        weights = None
        if self.attn == 'qkv-la':
            '''
            label-wise attention qkv
            return: batch_size(16) * class_num(52) * hidden_size(768)
            https://aclanthology.org/2021.emnlp-main.253.pdf
            '''
            b, _, _ = last_hidden_state.shape
            label_feat = label_feat.unsqueeze(0).repeat(b, 1, 1)
            attn_output, weights = self.attn_net(label_feat, last_hidden_state, last_hidden_state)
        elif self.attn == 'la':
            '''
            label-wise attention
            return: batch_size(16) * class_num(52) * hidden_size(768)
            https://aclanthology.org/2020.emnlp-main.235.pdf
            '''
            weights = self.tanh(self.linear1(last_hidden_state))  # (batch_size, seq_length, hidden_size)
            b, _, _ = last_hidden_state.shape
            label_feat = label_feat.unsqueeze(0).repeat(b, 1, 1)  # (batch_size, class_num, hidden_size)
            weights = torch.bmm(label_feat, weights.permute(0, 2, 1))
            weights = nn.Softmax(dim=1)(weights)  # (batch_size, class_num, seq_length)
            attn_output = torch.bmm(weights, last_hidden_state)  # (batch_size, class_num, hidden_size)
        elif self.attn == 'sa':
            '''
            self-attention
            return: batch_size * hidden_size (768)
            https://arxiv.org/pdf/1703.03130.pdf
            '''
            b = last_hidden_state.shape[0]
            vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)
            h = self.tanh(self.linear1(last_hidden_state))  # (b, s, 256)
            weights = torch.bmm(h, vectors)  # (b, s, 4)
            weights = nn.Softmax(dim=1)(weights)  # (b, s, 4)
            outputs = torch.bmm(weights.permute(0, 2, 1), h).view(b, -1)  # (b, 4h)
            attn_output = self.linear2(outputs)
        return attn_output, weights

    def forward(self, ids=None, mask=None, meta_kn_ids=None, meta_kn_mask=None, G=None):
        last_hidden_states = None
        graph_feat = None
        meta_last_hidden_states = None
        out = None

        text_feat = None
        meta_text_feat = None
        # aggregate information from heterogeneous graph
        if self.graph_model:
            graph_feat = self.graph_model(G)  # class_num * hidden_size(768)

        if ids != None:
            last_hidden_states, pooled_output = self.bert(ids, mask)
            out = pooled_output

        if meta_kn_ids != None:
            meta_last_hidden_states, meta_pooled_output = self.bert(meta_kn_ids, meta_kn_mask)
            out = meta_pooled_output

        if self.graph_model:
            if last_hidden_states != None:
                text_feat, weights = self.attn_network(last_hidden_states, graph_feat['protocols'])
                out = text_feat
            if meta_last_hidden_states != None:
                meta_text_feat, meta_weights = self.attn_network(meta_last_hidden_states, graph_feat['protocols'])
                out = meta_text_feat

        if self.fusion != None:
            out = self._fusion(out, graph_feat['protocols'])

        if self.cls != None:
            out = self.fc(out.reshape(out.shape[0], -1))

        return out, text_feat, meta_text_feat, graph_feat



class classifier(nn.Module):
    def __init__(self, class_num, dp):
        # G1(0, 10]: 8 protocols
        # G2(10, 100]: 19 protocols
        # G4(100, inf): 11 protocols
        super(classifier, self).__init__()
        self.class_num = class_num
        self.hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(self.class_num * self.hidden_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dp),
            nn.Linear(256, self.class_num)
        )
        # self.activation = nn.Sigmoid()

    def forward(self, feats):
        out = self.fc(feats.reshape(feats.shape[0], -1))
        return out
        # return self.activation(out)




