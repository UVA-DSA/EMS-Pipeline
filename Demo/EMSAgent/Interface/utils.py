import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from EMSAgent.Interface.default_sets import dataset, SAVE_RESULT_ROOT
if dataset == 'EMS':
    from EMSAgent.Interface.default_sets import p_node, group_p_dict, reverse_group_p_dict, ungroup_p_node, group_hier, ungroup_hier, p2hier
elif dataset == 'MIMIC3':
    from EMSAgent.Interface.default_sets import ICD9_DIAG
import re
from collections import OrderedDict
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

cahcedStopwords = stopwords.words('english')

from nltk.data import find

# Function to check and download if necessary
def download_nltk_data(dataset_name):
    try:
        # Attempt to find the dataset
        if dataset_name == 'punkt':
            find(f"tokenizers/{dataset_name}")
        else:
            find(f"corpora/{dataset_name}")
            
        print(f"{dataset_name} is already available.")
    except LookupError:
        # If dataset is not found, download it
        print(f"Downloading {dataset_name}...")
        nltk.download(dataset_name, quiet=True)

# Check and download stopwords and punkt
download_nltk_data('stopwords')
download_nltk_data('punkt')

class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def preprocess(text):
    text_ = removePunctuation(text.lower())
    text_tokens = word_tokenize(text_)
    # text_tokens_stem = [self.stemmer.stem(w) for w in text_tokens]
    tokens_without_sw = [word for word in text_tokens if not word in cahcedStopwords]
    new_text = ' '.join(tokens_without_sw)
    return new_text

def convert_label(labels, ages, logits=None):
    '''
    labels: [[ohe], [ohe], [ohe]]
    logits: [[logit], [logit], [logit]]
    ages: [tensor, tensor, tensor]
    convert clustered labels to exact labels
    p_node contains protocol names (the sequence order the same with labels).
    '''

    encodings = []
    encoding_logits = []

    base_node = p_node
    refer_node = ungroup_p_node #[ungroup_p_node, ungroup_hier]
    convert_dict = reverse_group_p_dict #[reverse_group_p_dict, reverse_group_hier_dict]
    map_dict = group_p_dict

    # ages = [age.numpy() for age in ages]
    for i, label in enumerate(labels):
        label_name = [base_node[j] for j in range(len(label)) if label[j] == 1]
        label_name_indices = [j for j in range(len(label)) if label[j] == 1]
        # convert the name 2 according to another sequence
        encoding = [0] * len(refer_node)
        encoding_logit = [0] * len(refer_node)
        for k, n in enumerate(label_name):
            if n in refer_node:
                idx = refer_node.index(n)
            # use age to determine
            else:
                age = int(ages[i])
                ### if it's pediatric ###
                if age < 18:
                    p_name = convert_dict[n][1]
                    if p_name in refer_node:
                        idx = refer_node.index(p_name)
                    else:
                        idx = refer_node.index(convert_dict[n][0])
                ### if it's adult ###
                else:
                    p_name = convert_dict[n][0]
                    if p_name in refer_node:
                        idx = refer_node.index(p_name)
                    else:
                        idx = refer_node.index(convert_dict[n][1])
            if type(logits) == np.ndarray:
                encoding_logit[idx] = logits[i][label_name_indices[k]]
            encoding[idx] = 1
        encodings.append(np.array(encoding))
        
        ##### convert logits #####
        if type(logits) == list:
            for l in range(len(encoding_logit)):
                if encoding_logit[l] == 0:
                    if refer_node[l] in base_node:
                        q = base_node.index(refer_node[l])
                        encoding_logit[l] = logits[i][q]
                    else:
                        group_p_name = map_dict[refer_node[l]]
                        logit = logits[i][base_node.index(group_p_name)]
                        age = int(ages[i])
                        if age < 18:
                            if convert_dict[group_p_name][1] in refer_node:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][1])] = logit
                            else:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][0])] = logit
                        else:
                            if convert_dict[group_p_name][0] in refer_node:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][0])] = logit
                            else:
                                encoding_logit[refer_node.index(convert_dict[group_p_name][1])] = logit
        encoding_logits.append(np.array(encoding_logit))
    return encodings, encoding_logits

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


