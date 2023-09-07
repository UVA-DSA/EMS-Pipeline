import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from .default_sets import device, p_node, reverse_group_p_dict
import re
from collections import OrderedDict

class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]['value']
    def __setattr__(self, attr, value):
        self[attr] = value

# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self

def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
            tempered
            / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )
    else:
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)
    return tempered.cpu().numpy()


def convertListToStr(l):
    s = ""
    if len(l) == 0: return s
    for each in l:
        s += str(each) + ';'
    return s[:-1]


def checkOnehot(X):
    for x in X:
        if x != 0 and x != 1:
            return False
    return True

def sortby(p2tfidf):
    for p, values in p2tfidf.items():
        od = OrderedDict(sorted(values.items(), key=lambda x:x[1], reverse=True))
        p2tfidf[p] = od
    return p2tfidf

def ungroup(age, protocols):
    ungroup_protocols = []
    for p in protocols.split(';'):
        p = p.strip().lower()
        if p in reverse_group_p_dict:
            if int(age) <= 21:
                ungroup_protocols.append(reverse_group_p_dict[p][1])
            else:
                ungroup_protocols.append(reverse_group_p_dict[p][0])
        else:
            ungroup_protocols.append(p)
    return convertListToStr(ungroup_protocols)

def onehot2p(onehot):
    pred = []
    for i in range(len(onehot)):
        if onehot[i] == 1:
            p_name = p_node[i]
            pred.append(p_name)
    return convertListToStr(pred)

def onehot2logits(onehot, logits):
    prob = logits[np.where(onehot)]

def removePunctuation(sentence):
    sentence = re.sub(r'[?|!|\'|"|;|:|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)
    sentence = sentence.strip()
    sentence = sentence.replace("\n", " ")
    return sentence

def text_remove_double_space(text):
    text = text.lower()
    res = ''
    for word in text.split():
        res = res + word + ' '
    return res.strip()


def deterConf(preds, targets, path):
    logits = [t.cpu().detach().numpy().tolist() for t in preds]
    labels = [t.cpu().detach().numpy().tolist() for t in targets]

    pair = []
    unpair = []
    for i in range(len(logits)):
        index = np.where(np.array(labels[i]) == 1)[0]
        pred_index = np.where(np.array(logits[i]) > 0.5)[0]
        if str(index) == str(pred_index):
            pair.append((logits[i], labels[i]))
        else:
            unpair.append((logits[i], labels[i]))
    values = []
    for i in range(len(pair)):
        logits = pair[i][0]
        indices = np.where(np.array(logits) > 0.5)[0]
        value = [logits[idx] for idx in indices]
        values.extend(value)

    unvalues = []
    for i in range(len(unpair)):
        logits = unpair[i][0]
        indices = np.where(np.array(logits) > 0.5)[0]
        value = [logits[idx] for idx in indices]
        unvalues.extend(value)

    correct_counts, correct_bins = np.histogram(values, bins=5)
    incorrect_counts, incorrect_bins = np.histogram(unvalues, bins=5)
    # SET DEFAULT CONFIDENCE SCORE AS 0.9
    confidence_score = 0.9
    for i in range(len(correct_counts)):
        if correct_counts[i] / incorrect_counts[i] - 1 > 0.5:
            confidence_score = correct_bins[i]
            break
    return confidence_score


def anchor(model, backbone, G, df):
    tokenizers = BertTokenizer.from_pretrained(backbone, do_lower_Case=True)
    overviews = []
    names = []
    for i in range(len(default_sets.p_node)):
        for j in range(len(df)):
            name = df.Protocols[j] + ' (' + df['Protocol ID'][j] +')'
            name = name.strip().lower()

            if name in default_sets.group_p_dict:
                name = default_sets.group_p_dict[name]

            if name == default_sets.p_node[i] and name not in names:
                # overviews.append(df.Overview[j])
                names.append(name)

                inputs = tokenizers.__call__(df.Overview[j],
                                    None,
                                    add_special_tokens=True,
                                    max_length=256,
                                    padding="max_length",
                                    truncation=True,
                                    )

                ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(device)
                ids = ids.unsqueeze(0)
                mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(device)
                mask = mask.unsqueeze(0)
                _, feats, _ = model(ids, mask, None, None, G)
                overviews.append(feats)
                break

    overviews = torch.stack(overviews).squeeze()
    return overviews

if __name__ =='__main__':
    import pandas as pd
    from model import EMSMultiModel
    from Heterogeneous_graph import HeteroGraph
    from default_sets import device

    signs_df = pd.read_excel('./data/Protocol_Impression files/All Protocols Mapping.xlsx')
    impre_df = pd.read_excel('./data/Protocol_Impression files/Impression Protocol.xlsx')
    med_df = pd.read_excel('./data/Protocol_Impression files/Medication Protocol.xlsx')
    proc_df = pd.read_excel('./data/Protocol_Impression files/Procedure Protocol.xlsx')
    HGraph = HeteroGraph(backbone='bvanaken/CORe-clinical-outcome-biobert-v1', mode='group')
    graph = HGraph(signs_df, impre_df, med_df, proc_df)

    df = pd.read_excel('./data/Protocol_Impression files/All Protocols Mapping.xlsx')
    model = EMSMultiModel('bvanaken/CORe-clinical-outcome-biobert-v1',
                          512,
                          'la',
                          None,
                          None,
                          graph)
    model.to(device)
    anchor_feats = anchor(model, 'bvanaken/CORe-clinical-outcome-biobert-v1', graph, df)
    print(anchor_feats.shape)

