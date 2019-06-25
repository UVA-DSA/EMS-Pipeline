import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import EnglishStemmer # load the stemmer module from NLTK
#from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import pandas as pd

class RAAdata(object):
    def __init__(self,text,vital,inter):
        self.text = text
        self.vital = vital
        self.inter = inter
        
def load_RAA_data(path, cv = True): 
    df = pd.read_excel(path)
    interset = set()
    interdict = dict()
    narratives = df['Narrative']
    narratives = [i for i in narratives]
    inters = df['Interventions']
    vitals = df['Vitals']
    vitals = [i for i in vitals]
    interventions = []
    for item in inters:
        inter = item.strip('{}').split('}{')
        inter = [i.split(':')[-1].strip().lower() for i in inter]
        c_int = []
        for j in inter:
            interset.add(j)
            if j in interdict:
                interdict[j] += 1
            else:
                interdict[j] = 1
            c_int.append(j)
        interventions.append(c_int)
    for inter in list(interdict):
        if cv and interdict[inter] < 20: del interdict[inter]
    data = [RAAdata(item,vitals[idx],interventions[idx]) for idx,item in enumerate(narratives)]
    
    return data,interdict

def fullmatch(regex, string, flags=0):
    """Emulate python-3.4 re.fullmatch()."""
    return re.match("(?:" + regex + r")\Z", string, flags=flags)

# preprocess utils
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def weighted_precision_recall_f1_util (y_test, y_pre, weight = None):
    tp, fp, fn = [0. for _ in range(len(y_pre[0]))], [0. for _ in range(len(y_pre[0]))], \
    [0. for _ in range(len(y_pre[0]))]
    for idx in range(len(y_pre)):
        for i in range(len(y_pre[idx])):
            if y_pre[idx][i] == 1 and y_test[idx][i] == 1: tp[i] += 1
            elif y_pre[idx][i] == 1 and y_test[idx][i] == 0: fp[i] += 1
            elif y_pre[idx][i] == 0 and y_test[idx][i] == 1: fn[i] += 1
    precision = [tp[i] / (tp[i] + fp[i]) if tp[i] > 0 or fp[i] > 0 else 0. for i in range(len(tp))]
    recall = [tp[i] / (tp[i] + fn[i]) if tp[i] > 0 or fn[i] > 0 else 0. for i in range(len(tp))]
    f1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) \
         if precision[i] > 0 or recall[i] > 0 else 0. for i in range(len(tp))]
    return np.average(precision, weights = weight), np.average(recall, weights = weight), \
np.average(f1, weights = weight)

def weighted_precision (y_test, y_pre, weight = None):
    precision, _, _ = weighted_precision_recall_f1_util (y_test, y_pre, weight)
    return precision

def weighted_recall (y_test, y_pre, weight = None):
    _, recall, _ = weighted_precision_recall_f1_util (y_test, y_pre, weight)
    return recall

def weighted_f1 (y_test, y_pre, weight = None):
    _, _, f1 = weighted_precision_recall_f1_util (y_test, y_pre, weight)
    return f1

def show_results(scores):
    metrics = ['test_precision_weighted','test_recall_weighted', 'test_f1_weighted',\
            'test_precision_micro', 'test_recall_micro', 'test_f1_micro']
    for metric in metrics:
        print metric + ':' + '%.2f' % np.average(scores[metric])
        
def risk_factor(gt, probs, preds):
    risk = []
    for idx,case in enumerate(probs):
        r = 0
        for i,prob in enumerate(case):
            if preds[idx][i] == 1 and gt[idx][i] == 0:
                r += prob * int2fp_score[num2int[i]] / sum(gt[idx])
            if preds[idx][i] == 0 and gt[idx][i] == 1:
                r += prob * int2fn_score[num2int[i]] / sum(gt[idx])
        risk.append(r)
    return sum(risk) / len(risk)

def trans_prob(probs):
    transed_prob = [[0.] * len(probs) for _ in range(len(probs[0]))]
    for idx, res in enumerate(probs):
        for i, p in enumerate(res):
            if len(p) < 2: transed_prob[i][idx] = 1. - p[0]
            else: transed_prob[i][idx] = p[1]
                
    return transed_prob

def show_test_results(gt, res, prob, class_weight):
    print "precision_micro" + ':' + '%.2f' % precision_score(gt, res, average = 'micro')
    print "recall_micro" + ':' + '%.2f' % recall_score(gt, res, average = 'micro')
    print "f1_micro" + ':' + '%.2f' % f1_score(gt, res, average = 'micro')
    print "precision_weighted" + ':' + '%.2f' % weighted_precision(gt, res, class_weight)
    print "recall_weighted" + ':' + '%.2f' % weighted_recall(gt, res, class_weight)
    print "f1_weighted" + ':' + '%.2f' % weighted_f1(gt, res, class_weight)
    print "risk_factor" + ':' + '%.4f' % risk_factor(gt, prob, res)
    
def filtering(res, prob, threshold):
    for idx, case in enumerate(res):
        for i in range(len(case)):
            if prob[idx][i] < threshold:
                res[idx][i] = 0
                prob[idx][i] = 0.
    return res, prob
