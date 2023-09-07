import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from default_sets import dataset, SAVE_RESULT_ROOT
if dataset == 'EMS':
    from default_sets import device, p_node, reverse_group_p_dict, ungroup_p_node, group_hier, ungroup_hier, p2hier
elif dataset == 'MIMIC3':
    from default_sets import ICD9_DIAG
import re
import default_sets
from collections import OrderedDict, defaultdict
import os
import json
import pandas as pd


class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]['value']
    def __setattr__(self, attr, value):
        self[attr] = value

def cnt_instance_per_label(df):
    label_cnt = {}

    if dataset == 'EMS':
        column_name = 'Ungrouped Protocols'
    elif dataset == 'MIMIC3':
        column_name = 'ICD9_DIAG'
    else:
        raise Exception('check dataset in default_sets.py')

    for i in range(len(df)):
        if type(df[column_name][i]) == float:
            continue
        ps = df[column_name][i].strip()
        for p in ps.split(';'):
            p = p.strip()
            label_cnt[p] = label_cnt.get(p, 0) + 1
    return label_cnt




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

def p2onehot(ps, ref_list):
    '''
    :param ps: list of protocols, e.g.: ['medical - altered mental status (protocol 3 - 15)']
    :param ref_list: p_node, ungroup_p_node
    :return: onehot encoding
    '''
    ohe = [0] * len(ref_list)
    for p in ps:
        idx = ref_list.index(p)
        ohe[idx] = 1
    return ohe


def onehot2p(onehot):
    pred = []
    for i in range(len(onehot)):
        if onehot[i] == 1:
            p_name = p_node[i]
            pred.append(p_name)
    return convertListToStr(pred)

def removePunctuation(sentence):
    sentence = re.sub(r'[?|!|\'|"|;|:|#|&|-]', r' ', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/|_|~|<|>]', r' ', sentence)
    sentence = re.sub(r"[\([{})\]]", r' ', sentence)
    sentence = re.sub(r"[*]", r' ', sentence)
    sentence = re.sub(r"[%]", r' percentage', sentence)
    sentence = sentence.strip()
    sentence = sentence.replace("\n", " ")
    return sentence



def text_remove_double_space(text):
    text = text.lower()
    res = ''
    for word in text.split():
        res = res + word + ' '
    return res.strip()

'''
The following codes from https://github.com/MemoriesJ/KAMG/blob/6618c6633bbe40de7447d5ae7338784b5233aa6a/NeuralNLP-NeuralClassifier-KAMG/evaluate/classification_evaluate.py
'''

def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / k

def get_precision_at_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_precision_score(y_t, y_s, k=k))

    return np.mean(p_ks)

def ranking_recall_score(y_true, y_score, k=10):
    # https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf
    """Recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    return float(n_relevant) / n_pos

def get_recall_at_k(y_true, y_score, k=10):
    """Mean recall at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean recall @k : float
    """

    r_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            r_ks.append(ranking_recall_score(y_t, y_s, k=k))

    return np.mean(r_ks)

def ranking_rprecision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        return ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(k, n_pos)

def get_r_precision_at_k(y_true, y_score, k=10):
    """Mean precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    mean precision @k : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(ranking_rprecision_score(y_t, y_s, k=k))

    return np.mean(p_ks)

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def get_ndcg_at_k(y_true, y_predict_score, k, gains="exponential"):

    """Normalized discounted cumulative gain (NDCG) at rank k
        Parameters
        ----------
        y_true : array-like, shape = [n_samples]
            Ground truth (true relevance labels).
        y_predict_score : array-like, shape = [n_samples]
            Predicted scores.
        k : int
            Rank.
        gains : str
            Whether gains should be "exponential" (default) or "linear".
        Returns
        -------
        Mean NDCG @k : float
        """

    ndcg_s = []
    for y_t, y_s in zip(y_true, y_predict_score):
        if np.sum(y_t == 1):
            ndcg_s.append(ndcg_score(y_t, y_s, k=k, gains=gains))

    return np.mean(ndcg_s)


