from torch_geometric.nn import HGTConv, GATConv, GCNConv, Linear
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
from EMS_TinyBERT.default_sets import device, multi_graph, dataset
if dataset == 'EMS':
    from EMS_TinyBERT.default_sets import p_node, ungroup_p_node, EMS_DIR
elif dataset == 'MIMIC3':
    from EMS_TinyBERT.default_sets import ICD9_DIAG, ICD9_DIAG_GROUP, MIMIC_3_DIR


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

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        out = self.lin(x)
        if dataset =='EMS' and x.shape[0] > len(ungroup_p_node):
            return out[10:]
        if dataset == 'MIMIC3' and x.shape[0] > len(list(ICD9_DIAG.keys())):
            return out[18:]
        return out

class HGT(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        node_types = ['protocol', 'impression', 'medication', 'procedure']
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        meta_data = (['protocol', 'impression', 'medication', 'procedure'],
                     [('protocol', 'has', 'impression'),
                      ('protocol', 'suggests', 'medication'),
                      ('protocol', 'takes', 'procedure'),
                      ('impression', 'indicates', 'protocol'),
                      ('medication', 'is used by', 'protocol'),
                      ('procedure', 'is taken by', 'protocol')]
                     )
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, meta_data, num_heads, group='sum')
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
        return graph_feat['protocols']


class EMSMultiModel(nn.Module):
    def __init__(self, backbone, max_len, attn, cluster, fusion, cls, graph_type):
        super(EMSMultiModel, self).__init__()
        self.backbone = backbone
        if self.backbone == 'stanford-crfm/BioMedLM':
            self.hidden_size = 2560
        elif self.backbone in ['UFNLP/gatortron-base', 'microsoft/biogpt']:
            self.hidden_size = 1024
        elif self.backbone in ['google/mobilebert-uncased', 'nlpie/clinical-mobilebert', 'nlpie/bio-mobilebert']:
            self.hidden_size = 512
        elif self.backbone == 'nlpie/tiny-clinicalbert':
            self.hidden_size = 312
        elif self.backbone == 'CNN':
            self.hidden_size = 200
        else:
            self.hidden_size = 768
            # if self.backbone == 'CNN' and graph_type == 'GCN':
            #     self.hidden_size = 200
            # else:
            #     self.hidden_size = 768

        self.cluster = cluster
        if dataset == 'EMS':
            if self.cluster == 'group':
                self.num_class = len(p_node)
            elif self.cluster == 'ungroup':
                self.num_class = len(ungroup_p_node)
            else:
                raise Exception('mode can only be [group, ungroup]')
        elif dataset == 'MIMIC3':
            if self.cluster == 'ungroup':
                self.num_class = len(list(ICD9_DIAG.keys()))
                # self.num_class = 252
            elif self.cluster == 'group':
                self.num_class = len(ICD9_DIAG_GROUP)
            else:
                raise Exception('mode can only be [group, ungroup]')
        else:
            raise Exception('check dataset in default_sets.py')

        # self.dropout = nn.Dropout(p=0.5)
        self.dropout_embed = nn.Dropout(p=0.2)
        self.fusion = fusion
        self.seq_length = max_len
        self.attn = attn
        self.cls = cls
        self.graph_type = graph_type

        if self.backbone == 'CNN':
            ### note: first element is vocab size, 28996 is BERT tokenizer's vocab length ###
            # self.embed = nn.Embedding(147976, self.hidden_size)
            fname = '%s/Word2Vec.npy' % MIMIC_3_DIR if dataset == 'MIMIC3' else '%s/Word2Vec.npy' % EMS_DIR
            word2vec = np.load(fname)
            weights = torch.FloatTensor(word2vec)
            self.embed = nn.Embedding.from_pretrained(weights)

            if self.graph_type:
                self.kernel_size = 10
                self.num_kernel = 200
                self.textEncoder = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.num_kernel,
                                             kernel_size=self.kernel_size, padding=self.kernel_size // 2)
            else:
                filter_sizes = [3, 4, 5]
                num_filters = [100, 100, 100]
                self.num_kernel = 300
                self.textEncoder = nn.ModuleList([
                    nn.Conv1d(in_channels=self.hidden_size,
                              out_channels=num_filters[i],
                              kernel_size=filter_sizes[i])
                    for i in range(len(filter_sizes))
                ])
        else:
            self.textEncoder = AutoModel.from_pretrained(self.backbone, output_hidden_states=True) #BioBERT: bert; textEncoder


        # when construct graph class, only HGT needs parameter graph
        if self.graph_type == 'hetero':
            # change here
            self.graph_model = HGT(hidden_channels=256, out_channels=self.hidden_size, num_heads=8, num_layers=1)
            self.graph_model.to(device)
        elif self.graph_type == 'GCN':
            hidden_dim = 200
            self.graph_model = GCN(hidden_channels=hidden_dim, out_channels=self.hidden_size)
            self.graph_model.to(device)
            if multi_graph:
                self.graph_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        else:
            self.graph_model = None

        if self.attn == 'qkv-la':
            if self.backbone == 'CNN':
                self.attn_net = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True,
                                                      kdim=self.num_kernel, vdim=self.num_kernel)
            else:
                self.attn_net = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8, batch_first=True)
        elif self.attn == 'sa':
            self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)
            self.linear1 = nn.Linear(self.hidden_size, 256)
            self.linear2 = nn.Linear(4 * 256, self.hidden_size)
            self.tanh = nn.Tanh()
        elif self.attn == 'la':
            if self.backbone == 'CNN':
                self.linear1 = nn.Linear(self.num_kernel, self.hidden_size)
            else:
                self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.tanh = nn.Tanh()

        else:
            print('make sure in config attn=None')

        if self.cls != None:
            if self.attn != None:
                if self.graph_type == 'GCN':
                    if self.backbone == "CNN":
                        input_dim = self.num_class * (self.hidden_size + self.num_kernel) #graph_feats, text_feats
                    else:
                        input_dim = self.num_class * self.hidden_size * 2
                else:
                    if self.backbone == "CNN" and self.attn != 'qkv-la':
                        ## change here
                        input_dim = self.num_class * self.num_kernel
                        # input_dim = self.num_class * (self.hidden_size + self.num_kernel)
                    else:
                        input_dim = self.num_class * self.hidden_size ### change here
            else:
                if self.backbone == "CNN":
                    input_dim = self.num_kernel
                else:
                    input_dim = self.hidden_size
            self.fc = nn.Linear(input_dim, self.num_class)

    def attn_network(self, last_hidden_state, label_feat):
        attn_output = None
        weights = None
        if self.attn == 'qkv-la':
            '''
            label-wise attention qkv
            return: batch_size(16) * class_num(52) * hidden_size(768)
            https://aclanthology.org/2021.emnlp-main.253.pdf
            '''
            b, _, _ = last_hidden_state.shape # (batch_size, seq_length, hidden_size)
            label_feat = label_feat.unsqueeze(0).repeat(b, 1, 1) # (batch_size, class_num, hidden_size)
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

    def forward(self, ids=None, mask=None, G=None):
        last_hidden_states = None
        graph_feat = None
        out = None
        out_fine_grain = None
        text_feat = None
        # aggregate information from heterogeneous graph
        if self.graph_type:
            ### multi-graph
            if multi_graph:
                graph_feat1 = self.graph_model(G[0])
                graph_feat2 = self.graph_model(G[1])
                graph_feat3 = self.graph_model(G[2])
                graph_feat = torch.cat((graph_feat1, graph_feat2, graph_feat3), dim=-1)
                graph_feat = self.graph_linear(graph_feat)
            else:
                graph_feat = self.graph_model(G)  # class_num * hidden_size(768)

        if ids != None:
            if self.backbone == 'CNN':
                text_embedding = self.dropout_embed(self.embed(ids)).transpose(1, 2)  # (batch size, hidden_size, seq_len)
                # text_embedding = self.embed(ids).transpose(1, 2)
                if self.graph_type:
                    #### refer to https://github.com/MemoriesJ/KAMG/blob/6618c6633bbe40de7447d5ae7338784b5233aa6a/NeuralNLP-NeuralClassifier-KAMG/model/classification/zagcnn.py#L6
                    last_hidden_states = torch.relu_(self.textEncoder(text_embedding)).transpose(1, 2)
                    out = last_hidden_states
                else:
                    x_conv_list = [F.relu(conv1d(text_embedding)) for conv1d in self.textEncoder]
                    x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                                   for x_conv in x_conv_list]
                    last_hidden_states = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                                     dim=1)
                    # last_hidden_states = self.dropout(last_hidden_states)
                    out = last_hidden_states
            else:
                # transformer-based models
                output = self.textEncoder(input_ids=ids, attention_mask=mask)  #BioBERT: bert; textEncoder
                if self.backbone in ['stanford-crfm/BioMedLM', 'microsoft/biogpt']:
                    last_hidden_states = output[0]
                    out = last_hidden_states
                elif self.backbone in ['distilbert-base-uncased', 'nlpie/clinical-distilbert']:
                    last_hidden_states = output[0]
                    pooled_output = last_hidden_states[:, 0]
                    out = pooled_output
                else:
                    last_hidden_states, pooled_output = output[0], output[1]
                    out = pooled_output

        if self.graph_type:
            if last_hidden_states != None:
                if self.graph_type == 'GCN':
                    start = 10 if dataset == "EMS" else 18
                    if multi_graph:
                        #### multi-graph share the same node embedding
                        text_feat, weights = self.attn_network(last_hidden_states, G[0].x[start:])
                    else:
                        text_feat, weights = self.attn_network(last_hidden_states, G.x[start:])
                else:
                    # text_feat, weights = self.attn_network(last_hidden_states, G.x_dict['protocol'])
                    # change here
                    text_feat, weights = self.attn_network(last_hidden_states, graph_feat)

                out = text_feat

        if self.fusion != None:
            out = self._fusion(out, graph_feat)

        if self.cls != None:
            ### if this is a GPT model and not using Label-wise attention
            if self.backbone in ['stanford-crfm/BioMedLM', 'microsoft/biogpt'] and G == None:
                logits = self.fc(out)
                b = logits.shape[0]
                out_fine_grain = logits[torch.arange(b, device=logits.device), -1]
            else:
                out = out.reshape(out.shape[0], -1)
                out_fine_grain = self.fc(out)

        return out_fine_grain, text_feat, graph_feat
