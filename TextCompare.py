import numpy as np
import re
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# hyper parameters
stop_words = set(stopwords.words('english'))

# compare two text
class TextComp(object):
    def __init__(self,original_path, recognition_path, encoding = 'utf-8'):
        # original_path: path of the original text
        # recognition_path: path of the recognized text
        # encoding: specifies the encoding which is to be used for the file
        self.original_path = original_path
        self.recognition_path = recognition_path
        self.encoding = encoding
        self.I = 0
        self.S = 0
        self.D = 0
    
    def Preprocess(self, path):
        with codecs.open(path, encoding = self.encoding) as f:
            text = f.read().lower()
            tokenizer = RegexpTokenizer(r'\w+')
            words = tokenizer.tokenize(text)
            filtered_words = filter(lambda w: w not in stop_words, words)
            return filtered_words
    
    def WER(self, debug = False):
        r = self.Preprocess(self.original_path)
        h = self.Preprocess(self.recognition_path)
        #costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3
     
        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r) + 1):
            costs[i][0] = i
            backtrace[i][0] = OP_DEL
         
        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = j
            backtrace[0][j] = OP_INS
     
        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i-1] == h[j-1]:
                    costs[i][j] = costs[i-1][j-1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i-1][j-1] + 1 # penalty is always 1
                    insertionCost    = costs[i][j-1] + 1   # penalty is always 1
                    deletionCost     = costs[i-1][j] + 1   # penalty is always 1
                 
                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL
                 
        # back trace though the best route:
        i = len(r)
        j = len(h)
        self.S = 0
        self.D = 0
        self.I = 0
        numCor = 0
        if debug:
            print("OP\toriginal\trecognition")
            lines = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("OK\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_SUB:
                self.S += 1
                i -= 1
                j -= 1
                if debug:
                    lines.append("SUB\t" + r[i]+"\t"+h[j])
            elif backtrace[i][j] == OP_INS:
                self.I += 1
                j -= 1
                if debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
            elif backtrace[i][j] == OP_DEL:
                self.D += 1
                i -= 1
                if debug:
                    lines.append("DEL\t" + r[i]+"\t"+"****")
        if debug:
            lines = reversed(lines)
            for line in lines:
                print(line)
            print("#cor " + str(numCor))
            print("#sub " + str(self.S))
            print("#del " + str(self.D))
            print("#ins " + str(self.I))
            return (self.S + self.D + self.I) / float(len(r))
        wer_result = round( (self.S + self.D + self.I) / float(len(r)), 3)
        return wer_result
    
    def Accuracy(self):
        return float(len(self.Preprocess(self.original_path)) - self.D - self.S)/len(self.Preprocess(self.original_path))
    