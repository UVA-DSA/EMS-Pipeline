import re

from torch_geometric.data import HeteroData, Data
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from .utils import sortby
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from .default_sets import p_node, group_p_dict, ungrouped_p_node, device, hier, p2hier
import os
import json

class bertEmbedding():
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = 256

    def tokenization(self, sentence):
        ids, segs = [], []
        inputs = self.tokenizer.__call__(sentence,
                                         None,
                                         add_special_tokens=True,
                                         max_length=self.max_len,
                                         padding="max_length",
                                         truncation=True)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    def fusion(self, token_embeddings, mode):
        if mode == 'SUM':
            '''
            sum up last 4 embeddings of biobert, and take 
            the mean in each dimension to get sentence embedding
            '''
            token_vecs_sum = []
            for token in token_embeddings:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec.numpy())
            return np.array(token_vecs_sum).mean(axis=0)
        elif mode == 'MEAN':
            '''
            take the second 2 last layer and then take the mean
            '''
            token_vecs = token_embeddings[:, -2, :]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            return sentence_embedding.numpy()
        else:
            raise Exception("check embedding mode")

    def getPreEmbedding(self, node, method='SUM'):
        ids, segs = self.tokenization(node)
        self.model.eval()
        with torch.no_grad():
            output = self.model(ids, segs)
        token_embeddings = torch.stack(output.hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        sentence_embedding = self.fusion(token_embeddings, method)
        return sentence_embedding.tolist()

class HeteroGraph(nn.Module):
    def __init__(self, backbone, mode):
        super(HeteroGraph, self).__init__()
        if mode == 'group':
            self.p_node = p_node #[hier, p_node]
        elif mode == 'ungroup':
            self.p_node = ungrouped_p_node
        else:
            raise Exception('mode can only be [group, ungroup]')
        self.group_p_dict = group_p_dict #[p2hier, group_p_dict]
        self.sign_node = []
        self.med_node = []
        self.proc_node = []
        self.p2idx = {}
        self.s2idx = {}
        self.m2idx = {}
        self.proc2idx = {}
        self.backbone = backbone
        self.mode = mode
        self.device = device

    @staticmethod
    def p2overview(signs_df, mode, group_p_dict):
        # create mapping (protocol --- overview)
        p2overview = defaultdict(list)
        for i in range(len(signs_df)):
            overview = signs_df['Overview'][i]
            if type(overview) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()

            if mode == 'group':
                if pname in group_p_dict:
                    pname = group_p_dict[pname]

            p2overview[pname].append(overview)
        return p2overview

    # the following mapping relations are generated from protocol guidelines
    @staticmethod
    def p2signs(signs_df, mode, group_p_dict):
        # create mapping (protocol --- signs)
        p2s = defaultdict(list)
        p2s_d = defaultdict(dict)
        s2p = defaultdict(list)
        for i in range(len(signs_df)):
            ss = signs_df['Signs and Symptoms(in impression list)'][i]
            if type(ss) == float:
                continue
            pname = signs_df['Protocols'][i] + ' ({})'.format(signs_df['Protocol ID'][i])
            pname = pname.lower()

            if mode == 'group':
                if pname in group_p_dict:
                    pname = group_p_dict[pname]

            for s in ss.split(';'):
                s = s.strip().capitalize()
                p2s_d[pname][s] = p2s_d[pname].get(s, 0) + 1
                if pname not in s2p[s]:
                    s2p[s].append(pname)
                if s not in p2s[pname]:
                    p2s[pname].append(s)
        return p2s, s2p, p2s_d

    @staticmethod
    def p2impre(impre_df, mode, group_p_dict):
        p2impre = defaultdict(list)
        p2impre_d = defaultdict(dict)
        impre2p = defaultdict(list)
        for i in range(len(impre_df)):
            ps = impre_df['Protocol_name'][i]
            candi_ps = impre_df['Indication/Considerations of Protocols'][i]
            impre = impre_df['Impression'][i].strip().capitalize()
            if type(impre) == float:
                continue
            if type(ps) != float:
                for p in ps.split(';'):
                    p = p.strip().lower()
                    if mode == 'group':
                        if p in group_p_dict:
                            p = group_p_dict[p]

                    p2impre_d[p][impre] = p2impre_d[p].get(impre, 0) + 1
                    if impre not in p2impre[p]:
                        p2impre[p].append(impre)
                    if p not in impre2p[impre]:
                        impre2p[impre].append(p)
            if type(candi_ps) != float:
                for c_p in candi_ps.split(';'):
                    c_p = c_p.strip().lower()

                    if mode == 'group':
                        if c_p in group_p_dict:
                            c_p = group_p_dict[c_p]

                    p2impre_d[c_p][impre] = p2impre_d[c_p].get(impre, 0) + 1
                    if impre not in p2impre[c_p]:
                        p2impre[c_p].append(impre)
                    if c_p not in impre2p[impre]:
                        impre2p[impre].append(c_p)
        return p2impre, impre2p, p2impre_d

    @staticmethod
    def p2med(med_df, mode, group_p_dict):
        # create mapping (protocol --- medication)
        p2m = defaultdict(list)
        p2m_d = defaultdict(dict)
        m2p = defaultdict(list)
        med_filter = ['Oxygen (7806)', 'Normal saline (125464)']
        for i in range(len(med_df)):
            med_name = med_df['medication'][i].strip().capitalize()
            if med_name in med_filter:
                continue

            med_name_idx = re.search(r'[(]\d+[)]', med_name).start()
            med_name = med_name[:med_name_idx].strip()
            pname = ''
            if type(med_df['protocols'][i]) != float:
                pname += med_df['protocols'][i] + '; '
            if type(med_df['considerations'][i]) != float:
                pname += med_df['considerations'][i]
            if type(pname) != float:
                pname = pname.lower()
            for p in pname.split(';'):
                p = p.lower().strip()
                if mode == 'group':
                    if p in group_p_dict:
                        p = group_p_dict[p]

                if p == '':
                    continue
                p2m_d[p][med_name] = p2m_d[p].get(med_name, 0) + 1
                if med_name not in p2m[p]:
                    p2m[p].append(med_name)
                if p not in m2p[med_name]:
                    m2p[med_name].append(p)
        return p2m, m2p, p2m_d

    @staticmethod
    def p2proc(proc_df, mode, group_p_dict):
        # create mapping (protocol --- procedure)
        p2proc = defaultdict(list)
        p2proc_d = defaultdict(dict)
        proc2p = defaultdict(list)
        proc_filter = ['Assess airway', 'Ecg', 'Iv', 'Move patient', 'Bvm', 'Assess - pain assessment',
                       'Im/sq injection']
        for i in range(len(proc_df)):
            procedure = proc_df['Grouping'][i].strip().capitalize()
            if procedure in proc_filter:
                continue

            if type(proc_df['Protocols'][i]) == float:
                continue
            for p in proc_df['Protocols'][i].split(';'):
                p = p.lower().strip()

                if mode == 'group':
                    if p in group_p_dict:
                        p = group_p_dict[p]

                p2proc_d[p][procedure] = p2proc_d[p].get(procedure, 0) + 1
                if procedure not in p2proc[p]:
                    p2proc[p].append(procedure)
                if p not in proc2p[procedure]:
                    proc2p[procedure].append(p)
        return p2proc, proc2p, p2proc_d

    # the following mapping relations are generated from RAA dataset
    @staticmethod
    def p2impre_data(mode):
        root = './json files/TF-IDF/{}'.format(mode)
        with open(os.path.join(root, 'impre2p.json'), 'r') as f:
            impre2p = json.load(f)

        with open(os.path.join(root, 'p2impre.json'), 'r') as f:
            p2impre_d = json.load(f)
        p2impre = {}
        for k, v in p2impre_d.items():
            p2impre[k] = list(v.keys())
        return p2impre, impre2p, p2impre_d

    @staticmethod
    def p2med_data(mode):
        root = './json files/TF-IDF/{}'.format(mode)
        with open(os.path.join(root, 'med2p.json'), 'r') as f:
            m2p = json.load(f)

        with open(os.path.join(root, 'p2med.json'), 'r') as f:
            p2m_d = json.load(f)
        p2m = {}
        for k, v in p2m_d.items():
            p2m[k] = list(v.keys())
        return p2m, m2p, p2m_d

    @staticmethod
    def p2proc_data(mode):
        root = './json files/TF-IDF/{}'.format(mode)
        with open(os.path.join(root, 'proc2p.json'), 'r') as f:
            proc2p = json.load(f)

        with open(os.path.join(root, 'p2proc.json'), 'r') as f:
            p2proc_d = json.load(f)
        p2proc = {}
        for k, v in p2proc_d.items():
            p2proc[k] = list(v.keys())
        return p2proc, proc2p, p2proc_d

    def _genNode(self, p2overview, s2p, m2p, proc2p):
        for k, v in s2p.items():
            # check if sign-mapped protocols is in p_node
            for i in v:
                if self.mode == 'group':
                    if i in group_p_dict:
                        i = group_p_dict[i]
                if i in self.p_node and k not in self.sign_node:
                    self.sign_node.append(k)
                    break

        for k, v in m2p.items():
            # check if medicine-mapped protocols is in p_node
            for i in v:
                if self.mode == 'group':
                    if i in group_p_dict:
                        i = group_p_dict[i]
                if i in self.p_node and k not in self.med_node:
                    self.med_node.append(k)
                    break

        for k, v in proc2p.items():
            # check if procdure-mapped protocols is in p_node
            for i in v:
                if self.mode == 'group':
                    if i in group_p_dict:
                        i = group_p_dict[i]
                if i in self.p_node and k not in self.proc_node:
                    self.proc_node.append(k)
                    break
        node = []
        node.extend(self.p_node)
        node.extend(self.sign_node)
        node.extend(self.med_node)
        node.extend(self.proc_node)

        nodes_attr = []
        tokenizer = BertTokenizer.from_pretrained(self.backbone, do_lower_Case=True)
        model = BertModel.from_pretrained(self.backbone, output_hidden_states=True)
        b_embed = bertEmbedding(tokenizer, model)
        q, w, e, r = 0, 0, 0, 0
        for i, n in enumerate(tqdm(node, desc='Generate Node Embedding')):
            attr = {}
            if n in self.p_node:
                n_type = 'protocol'
                n_features = []
                for view in p2overview[n]:
                    n_features.append(b_embed.getPreEmbedding(view))
                n_feature = list(np.mean(n_features, axis=0))
                self.p2idx[i] = q
                q += 1
            elif n in self.sign_node:
                n_type = 'sign'
                n_feature = b_embed.getPreEmbedding(n)
                self.s2idx[i] = w
                w += 1
            elif n in self.med_node:
                n_type = 'medication'
                n_feature = b_embed.getPreEmbedding(n)
                self.m2idx[i] = e
                e += 1
            elif n in self.proc_node:
                n_type = 'procedure'
                n_feature = b_embed.getPreEmbedding(n)
                self.proc2idx[i] = r
                r += 1
            else:
                raise Exception('Node type is incorrect, recheck node type')
            attr[n_type] = n
            attr['node_type'] = n_type
            attr['node_feature'] = n_feature
            nodes_attr.append((i, attr))
        return node, nodes_attr

    def __checkEdges(self, node, src_node, dic):
        edges = []
        if src_node in dic:
            for i, dst_node in enumerate(dic[src_node]):
                src_node_idx = node.index(src_node)
                dst_node_idx = node.index(dst_node)
                # src_node_type = 'protocol'
                # dst_node_type = 'signs'
                if dst_node in self.sign_node:
                    edge_type = 'has'
                # dst_node_type = 'medication'
                elif dst_node in self.med_node:
                    edge_type = 'suggests'
                # dst_node_type = 'procedure'
                elif dst_node in self.proc_node:
                    edge_type = 'takes'
                else:
                    raise Exception('recheck edge type')
                edges.append((src_node_idx, dst_node_idx, {'edge_type': edge_type}))
        return edges

    def _genEdge(self, node, p2s, p2m, p2proc):
        edges = []
        for i, n in enumerate(node):
            edge_s = self.__checkEdges(node, n, p2s)
            if edge_s:
                edges.extend(edge_s)
            edge_m = self.__checkEdges(node, n, p2m)
            if edge_m:
                edges.extend(edge_m)
            edge_proc = self.__checkEdges(node, n, p2proc)
            if edge_proc:
                edges.extend(edge_proc)
        return edges

    @staticmethod
    def transIdx(idx, protocol_idx_map, sign_idx_map, med_idx_map, proc_idx_map):
        if idx in protocol_idx_map:
            return protocol_idx_map[idx], 'protocol'
        elif idx in sign_idx_map:
            return sign_idx_map[idx], 'sign'
        elif idx in med_idx_map:
            return med_idx_map[idx], 'medication'
        elif idx in proc_idx_map:
            return proc_idx_map[idx], 'procedure'
        else:
            raise Exception('node indexing is incorrect')

    def _genGraph(self, nodes_attr, edges):
        # convert graph data into class HeteroData
        label_hetero = HeteroData()

        # add node features
        p_feat = []
        s_feat = []
        m_feat = []
        proc_feat = []

        for (index, node_attr) in nodes_attr:
            if node_attr['node_type'] == 'protocol':
                p_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'sign':
                s_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'medication':
                m_feat.append(node_attr['node_feature'])
            elif node_attr['node_type'] == 'procedure':
                proc_feat.append(node_attr['node_feature'])
            else:
                raise Exception('incorrect node type')

        label_hetero['protocol'].x = torch.tensor(p_feat, dtype=torch.float).to(self.device)
        label_hetero['impression'].x = torch.tensor(s_feat, dtype=torch.float).to(self.device)
        label_hetero['medication'].x = torch.tensor(m_feat, dtype=torch.float).to(self.device)
        label_hetero['procedure'].x = torch.tensor(proc_feat, dtype=torch.float).to(self.device)

        phs_src, phs_dst = [], []
        psm_src, psm_dst = [], []
        ptp_src, ptp_dst = [], []
        for (src, dst, _) in edges:
            src_idx, src_name = self.transIdx(src, self.p2idx, self.s2idx, self.m2idx, self.proc2idx)
            dst_idx, dst_name = self.transIdx(dst, self.p2idx, self.s2idx, self.m2idx, self.proc2idx)
            if dst_name == 'sign':
                phs_src.append(src_idx)
                phs_dst.append(dst_idx)
            elif dst_name == 'medication':
                psm_src.append(src_idx)
                psm_dst.append(dst_idx)
            elif dst_name == 'procedure':
                ptp_src.append(src_idx)
                ptp_dst.append(dst_idx)
            else:
                raise Exception('check transIdx')

        label_hetero['protocol', 'has', 'impression'].edge_index = torch.tensor([phs_src, phs_dst]).to(self.device)
        label_hetero['protocol', 'suggests', 'medication'].edge_index = torch.tensor([psm_src, psm_dst]).to(self.device)
        label_hetero['protocol', 'takes', 'procedure'].edge_index = torch.tensor([ptp_src, ptp_dst]).to(self.device)

        label_hetero['impression', 'indicates', 'protocol'].edge_index = torch.tensor([phs_dst, phs_src]).to(self.device)
        label_hetero['medication', 'is used by', 'protocol'].edge_index = torch.tensor([psm_dst, psm_src]).to(self.device)
        label_hetero['procedure', 'is taken by', 'protocol'].edge_index = torch.tensor([ptp_dst, ptp_src]).to(self.device)
        return label_hetero

    def forward(self, signs_df, impre_df, med_df, proc_df):
        p2overview = self.p2overview(signs_df, self.mode, self.group_p_dict)
        # p2s, s2p, _ = self.p2signs(signs_df, self.mode, self.group_p_dict)
        p2impre, impre2p, _ = self.p2impre(impre_df, self.mode, self.group_p_dict)
        p2m, m2p, _ = self.p2med(med_df, self.mode, self.group_p_dict)
        p2proc, proc2p, _ = self.p2proc(proc_df, self.mode, self.group_p_dict)

        # p2impre, impre2p, _ = self.p2impre_data(self.mode)
        # p2m, m2p, _ = self.p2med_data(self.mode)
        # p2proc, proc2p, _ = self.p2proc_data(self.mode)

        node, nodes_attr = self._genNode(p2overview, impre2p, m2p, proc2p)
        edges = self._genEdge(node, p2impre, p2m, p2proc)
        graph = self._genGraph(nodes_attr, edges)
        return graph

if __name__ == '__main__':
    ##### define the sequence of protocol as p_node
    # in total 44 grouped protocols
    signs_df = pd.read_excel('./data/Protocol_Impression files/All Protocols Mapping.xlsx')
    impre_df = pd.read_excel('./data/Protocol_Impression files/Impression Protocol.xlsx')
    med_df = pd.read_excel('./data/Protocol_Impression files/Medication Protocol.xlsx')
    proc_df = pd.read_excel('./data/Protocol_Impression files/Procedure Protocol.xlsx')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HGraph = HeteroGraph(backbone='bvanaken/CORe-clinical-outcome-biobert-v1', mode='group')
    G = HGraph(signs_df, impre_df, med_df, proc_df)
    # HGraph.gen_TF_IDF(signs_df, impre_df, med_df, proc_df)
    print(G)