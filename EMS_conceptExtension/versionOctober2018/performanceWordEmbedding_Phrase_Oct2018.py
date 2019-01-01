import gensim, logging
import io
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from itertools import repeat
import string
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize
import numpy as np
from numpy import dot
from numpy.linalg import norm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def isNumber(inputString):
    try:
        float(inputString)
        return True
    except ValueError:
        return False

def hasStopwords(inputString):
    stops = loadStopWords()
    tokens = inputString.split(" ")
    for item in stops:
        if item in tokens:
            return True
    return False

def loadData(filepath, remove_stopwords=False):
##    dataList = list()
    dataList = [[] for x in range(11891)]
    rfile = open(filepath, "r")
    for line in rfile:
        line2 = line.lower().strip()
        line2 = re.sub("[^a-zA-Z0-9_]"," ", line2)
        innerList = line2.split()
        if remove_stopwords:
            stops = loadStopWords()
            innerList = [w for w in innerList if not w in stops]
        innerList = [w for w in innerList if not w.isdigit()]
        dataList.append(innerList)
    rfile.close()
    return dataList


def loadStringData(filepath):
##    dataList = list()
##    dataList = [[] for x in range(11891)]
    dataString = ""
    rfile = open(filepath, "r")
    for line in rfile:
        line2 = line.lower().strip()
        line2 = re.sub("[^a-zA-Z0-9_]"," ", line2)
        dataString += " "+ line2
    rfile.close()
    return dataString

def loadStringData2(filepath):
##    dataList = list()
##    dataList = [[] for x in range(11891)]
    dataString = ""
    rfile = open(filepath, "r")
    for line in rfile:
        line = line.lower()
        dataString += line
    rfile.close()
    return dataString


def loadStopWords():
    filePath = "/Users/sarahmasudpreum/Desktop/PythonTCode/general/stopwords.txt"
    rfile = open(filePath, "r")
    stopWords = list()
    for line in rfile:
        stopWords.append(line.strip())
    rfile.close()
    return stopWords

def loadOriginalConceptListPhrase(filepath):
    conceptList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        item = line.lower().strip()
        if item not in conceptList:
            conceptList.append(item)
    rfile.close()
    return conceptList

def loadExtendedConceptListPhrase(filepath):
    conceptList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        tokens = line.lower().strip().split("\t")
        for token in tokens:
            if token not in conceptList:
                conceptList.append(token)
    rfile.close()

    return conceptList

def loadOriginalConceptList(filepath):
    conceptList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        concept = line.split("\t")[0].lower().strip()
        items = concept.split(" ")
        for item in items:
            if item not in conceptList:
                conceptList.append(item)

    stops = loadStopWords()##set(stopwords.words("english"))
    conceptList = [w for w in conceptList if not w in stops]
    rfile.close()
    return conceptList


def loadSimpleListPhrase (filePath):
    rfile = open(filePath, "r")
    itemList = list()
##    table = str.maketrans({key: None for key in string.punctuation})
    for line in rfile:
##        line = re.sub("[^a-zA-Z0-9_]"," ", line)
        item = line.lower().strip()
        if item not in itemList:
            itemList.append(item)
    rfile.close()
    return itemList


def computeFrequency (gtConceptList, frequencyList, wordList,fileName=""):
    wordFrequencyList = list(repeat(0,len(gtConceptList)))
    for item in wordList:
        for i in range (0,len(gtConceptList)):
            gTerm = str(gtConceptList[i]).strip()
            item = item.strip()
##            if len(gTerm.split(" ")) > 1:
            if len(gTerm) > 1 and item.find(gTerm) is not -1: #or gTerm.find(item) is not -1:
                wordFrequencyList[i] = frequencyList[i]
            if len(gTerm) is 1:
                tokens = item.split(" ")
                for token in tokens:
                    if token.find(gTerm) is not -1 and gTerm.find(token) is not -1:
                        wordFrequencyList[i] = frequencyList[i]

    return wordFrequencyList

def computeFP (filePath, conceptList, gtConceptList):
    tpConceptList = list()
    for item in conceptList:
        for i in range (0,len(gtConceptList)):
            gTerm = str(gtConceptList[i]).strip()
            item = item.strip()
            if len(gTerm) > 1 and item.find(gTerm) is not -1: #or gTerm.find(item) is not -1:
                tpConceptList.append(item)
            if len(gTerm) is 1:
                tokens = item.split(" ")
                for token in tokens:
                    if token.find(gTerm) is not -1 and gTerm.find(token) is not -1:
                        tpConceptList.append(item)


    filteredConceptList = [w for w in conceptList if w not in tpConceptList]
    filteredConceptList = list(set(filteredConceptList))
    # print("len(filteredConceptList):\t"+str(len(filteredConceptList)))
    rfile = open(filePath, "r")
    frequencyList = list(repeat(0,len(filteredConceptList)))
    stops = loadStopWords()##set(stopwords.words("english"))

    for line in rfile:
        for i in range (0,len(filteredConceptList)):
            if len(filteredConceptList[i]) > 1 and line.find(filteredConceptList[i]) is not -1:
                frequencyList[i] = 1
            if len(filteredConceptList[i]) is 1:
                tokens = line.split(" ")
                for token in tokens:
                    if token.find(filteredConceptList[i]) is not -1 and filteredConceptList[i].find(token) is not -1:
                        frequencyList[i] = 1
    rfile.close()
    # print(frequencyList)
    return frequencyList

def computePrecision (TPlist, FPList):
    # print(TPlist)
    # print(FPList)
    t = sum(TPlist) + sum(FPList)
    p = float(0)
    if t>0:
        p = sum(TPlist)/float( t )
    return round( p,4 )

def computeRecall (TPlist, denom):
    p = sum(TPlist)/float(denom)
    return round( p,4 )



def score(seq1, seq2, model):
        '''prepares inputs for scoring'''
        seq1_word_list = word_tokenize(seq1.strip().lower())
        seq2_word_list = word_tokenize(seq2.strip().lower())
        return sim_score(seq1_word_list, seq2_word_list,model)

def sim_score(wordlist1, wordlist2,model):
        '''calculates average maximum similarity between two phrase inputs'''
        maxes = []
        for word in wordlist1:
                cur_max = 0
                for word2 in wordlist2:
                        if word == word2: #case where words are identical
                                sim = 1
                                cur_max = sim
                        elif word in model and word2 in model:
                                sim = model.similarity(word, word2) #calculate cosine similarity score for words
                                if sim > cur_max:
                                        cur_max = sim
                if cur_max != 0:
                        maxes.append(cur_max)
        if sum(maxes) == 0:
            return 0
        return float(sum(maxes)) / len(maxes)

def sim_sum_score(wordlist1, wordlist2,model):
    '''calculates similarity between two phrase inputs'''
    x1 = np.zeros(100)
    x2 = np.zeros(100)

    for word in wordlist1:
        if word in model:
            x1 = x1 + model[word]
    if len(wordlist1) > 0:
        x1 = x1/len(wordlist1)

    for word in wordlist2:
        if word in model:
            x2 = x2 + model[word]
    if len(wordlist2) > 0:
        x2 = x2/len(wordlist2)

    if norm(x1) == 0 or norm(x2) == 0:
        return 0

    sim = dot(x1, x2)/(norm(x1)*norm(x2))
    return sim


def sim_avg_score(wordlist1, wordlist2,model):
        avgdist = []
        for word in wordlist1:
            cur_dist = 0
            for word2 in wordlist2:
                if word == word2:
                    sim = 1
                    cur_dist += sim
                elif word in model and word2 in model:
                    sim = model.similarity(word, word2) #calculate cosine similarity score for words
                    cur_dist += sim
            if cur_dist != 0:
##                print(cur_dist)
                cur_dist = cur_dist/float(len(wordlist2))
                avgdist.append(cur_dist)
        if sum(avgdist) == 0:
            return 0
##	for item in avgdist:
##            print(item)
        return float(sum(avgdist)) / len(avgdist)


def model_most_similar_phrases(model, inputSpace, seedConcept, topn):
    candidateSet = set()
    scoreList = list(repeat(0,len(inputSpace)))
    for i in range(0,len(inputSpace)):
        ip = inputSpace[i]
        scoreList[i] = score(ip, seedConcept, model)
    finalList = list(zip(inputSpace, scoreList))
    finalList.sort(key=lambda x: x[1])
    finalList = finalList[-topn:]
    # print(finalList)
    candidateSet.update(set(finalList))

    temp = np.array(list(candidateSet))
    candidateList = list(temp[:,0] )
    scoreTupleList = list(candidateSet)
    # print("*******\n*******\n")
    # print(temp)
    # return list(temp[:,0] )
    return candidateList, scoreTupleList

def createFileName(modelName, n=25):
    filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/PhraseResults/withUnigram/" #withUnigram withoutUnigram
    if "RAA" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_RAA.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_RAA.txt"
    elif "Forum" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_Forum.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_Forum.txt"
    elif "I2B2" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_I2B2.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_I2B2.txt"
    elif "MIMIC3" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_MIMIC3.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_MIMIC3.txt"
    elif "EMS" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_EMS.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_EMS.txt"
    elif "Med" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_Med.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_Med.txt"
    elif "all" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_MedEMS.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_MedEMS.txt"
    elif "Google" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_Google.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_Goolge.txt"
    elif "Glove" in modelName:
        filePath1 = filePath + str(n)+ "candidateConcepts_Glove.txt"
        filePath2 = filePath + str(n)+ "candidateConceptsWDistance_Glove.txt"

    return [filePath1, filePath2]





logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

datafileRootPath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/wordEmbeddingModel/Data/"
dataFileName = "MedEMSallDataIn1.txt" #EMSallDataIn1 MedallDataIn1 MedEMSallDataIn1
# filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/trainingData/DataLines.txt"
filePath = datafileRootPath + dataFileName
dataString = loadStringData2(filePath)
# dataString = re.sub("[^a-zA-Z0-9]"," ", dataString)
##        tokens = line.split(" ")
##        tokens = [w for w in tokens if w.find(" ") is -1 and not w.isdigit() and not w in stops]
##        line = " ".join(tokens)
sent_tokenize_list = sent_tokenize(dataString)
print(str(len(sent_tokenize_list)))
# tokens = nltk.wordpunct_tokenize(dataString) #word_tokenize
tokens = word_tokenize(dataString) #word_tokenize
print(str(len(tokens)))

tokens = [w for w in tokens if not w.isdigit()]
unigramList = list(set(tokens))
stops = loadStopWords()

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/conceptList.txt" # newConceptList_Sept2018 conceptList
conceptList = loadOriginalConceptListPhrase(filePath)
conceptList = [w for w in conceptList if w.find(" ") is -1]

finder = BigramCollocationFinder.from_words(tokens)
bigram_measures = nltk.collocations.BigramAssocMeasures()
##sorted(finder.nbest(bigram_measures.raw_freq, 100))
scored = finder.score_ngrams(bigram_measures.raw_freq)
bigramSet = set(sorted(bigram for bigram, score in scored))
bigramList = list(bigramSet)
len(bigramList)
filtBigramList = list()
filtBigramList= [u+" "+v for u,v in bigramList if not u in stops and not v in stops and (u in conceptList or v in conceptList)]
len(filtBigramList)
filtBigramList= [w for w in filtBigramList if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(w)) == 0 ]
# filtBigramList= [w for w in filtBigramList if w.find("patient") is -1 and w.find(" pt") is -1]
len(filtBigramList)
filtBigramList = list(set(filtBigramList))

finder = TrigramCollocationFinder.from_words(tokens)
trigram_measures = nltk.collocations.TrigramAssocMeasures()
scored = finder.score_ngrams(trigram_measures.raw_freq)
trigramSet = set(sorted(trigram for trigram, score in scored))
trigramList = list(trigramSet)
len(trigramList)
filtTrigramList = list()
filtTrigramList= [u+" "+v+" "+w for u,v,w in trigramList if not u in stops and not v in stops and not w in stops and (u in conceptList or v in conceptList or w in conceptList)]
len(filtTrigramList)
filtTrigramList= [w for w in filtTrigramList if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(w)) == 0 ]
# filtTrigramList = [w for w in filtTrigramList if w.find("patient") is -1 and w.find(" pt") is -1]
##filtTrigramList= [u+" "+v+" "+w for u,v,w in trigramList if not u in stops and not w in stops and (u in conceptList or v in conceptList or w in conceptList)]
filtTrigramList = list(set(filtTrigramList))
len(filtTrigramList)


##load GT annotation

##load protocol concept list
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/conceptList.txt" # newConceptList_Sept2018 conceptList
conceptList = loadOriginalConceptListPhrase(filePath)
# print(len(conceptList))
##load metaMap extended concept list
##count ground truth in testData


# filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/trainingData/DataLines.txt"
# filePath = datafileRootPath + dataFileName
# sentenceList = loadData(filePath, True)
# min_count = 1
# vectorSize = 150
# windowSize = 10
# cbow_mean = 0
# downsampling = 1e-3

# model = gensim.models.Word2Vec(sentenceList, sg=0, min_count=min_count, size=vectorSize, window=windowSize, sample = downsampling)

candidateList = list()
inputSpace = set()
inputSpace.update(filtTrigramList)
inputSpace.update(filtBigramList)
# inputSpace.update(unigramList)
inputSpace = list(inputSpace)

# model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sarahmasudpreum/src/word2vecModels/GoogleNews-vectors-negative300.bin', binary=True)
# word2vec_RAA_Data_150_10 word2vec_Forum_Data_150_10 word2vec_MIMIC3_Data_150_10 word2vec_I2B2_Data_150_10
# word2vec_Med_Data_150_10 word2vec_EMS_Data_150_10 word2vec_MedEMS_Data_150_10
model = gensim.models.Word2Vec.load('/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/wordEmbeddingModel/Model/word2vec_MedEMS_Data_150_10.model')
vl = model.wv.vocab ##model.vocab (for google) model.wv.vocab
TOPN = 50
f1, f2 = createFileName("all", TOPN)
## RAA Forum MIMIC3 I2B2 Med EMS all Google Glove
wfile = open(f1, "w")
wfile2 = open(f2, "w")
# candidateList = list()

##count ground truth in testData
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/FormattedTags/allInOne.txt"
gtConceptList = loadSimpleListPhrase(filePath)
frequencyList = list(repeat(0,len(gtConceptList)))

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/MetaMapOutputsConciseP.txt" ## MetaMapOutputsConciseW MetaMapOutputs_FilteredST_ConciseW
rfile1 = open(filePath,"r")
mmWordList = list()
for line in rfile1:
    phrase = line.strip()
    if not phrase.isdigit():
        if phrase not in mmWordList:
            mmWordList.append(phrase)
rfile1.close()

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData.txt"
rfile = open(filePath, "r")
for line in rfile:
    for i in range (0,len(gtConceptList)):
        line = line.lower().strip()
        if len(gtConceptList[i]) > 1 and line.find(gtConceptList[i]) is not -1:
            frequencyList[i] += 1
            # print(gtConceptList[i] +"\t\t"+ line)
        if len(gtConceptList[i]) is 1:
            tokens = line.split(" ")
            for token in tokens:
                if token.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(token) is not -1:
                    frequencyList[i] += 1
                    # print(gtConceptList[i] +"\t\t"+ token)
                    # print(line)
rfile.close()

##create Output file
denom = sum(frequencyList)
globalConceptList = set()
globalConceptList.update(conceptList)

for concept in conceptList:
    candidateList = list()
    if concept in vl:
        candidateList, scoreTupleList = model_most_similar_phrases(model, inputSpace, concept, topn=TOPN)
        candidateList = [w for w in candidateList if not hasNumbers(w)]
        candidateList = [w for w in candidateList if not hasStopwords(w)]
        # candidateList = [w for w in candidateList if w.find("patient") is -1 or w.find(" pt") is -1 or w.find(" t ") is -1 or w.find(" s ") is -1]
        candidateList= [w for w in candidateList if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(w)) == 0 ]
        candidateList = list(set(candidateList))


        if len(candidateList) > 0:
            globalConceptList.update(set(candidateList))
            wfile.write(concept+"\t")
            wfile2.write(concept+"\t")
            for item in candidateList:
                if not isNumber(item) and item.find("numberToken") is -1:
                    wfile.write(str(item)+"\t")
                    for conceptScore in scoreTupleList:
                        if conceptScore[0].find(item) is -1 and item.find(conceptScore[0]) is -1:
                            wfile2.write(str(item)+","+str(conceptScore[1])+"\t")
                            # print(str(item)+","+str(conceptScore[1])+"\t")
                            break
        wfile.write("\n")
        wfile2.write("\n")

wfile.close()
wfile2.close()


filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/PhraseResults/withUnigram/results.txt" #withUnigram withoutUnigram
resultWriteFile = open(filePath, "a")
# resultWriteFile.write("\tRecall\tPrecision\n")
resultWriteFile.write("EMSMed+withoutUnigram\t\t\n")

# filePath ="/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/PhraseResults/withUnigram/25candidateConcepts_EMS.txt"
# # globalConceptList = list()
# temp = loadExtendedConceptListPhrase(filePath)
# globalConceptList = list(globalConceptList) + temp
globalConceptList = list(set(globalConceptList))
# globalConceptList.append(conceptList)

w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, globalConceptList)
testDatafilePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData.txt"
w2vFPList = computeFP(testDatafilePath, globalConceptList, gtConceptList)
w2vr = computeRecall(w2vFrequencyList, denom)
w2vp = computePrecision(w2vFrequencyList, w2vFPList)
resultWriteFile.write("withOutMetaMap\t"+str(w2vr)+"\t"+str(w2vp)+"\n")
print(str(w2vr)+"\t"+str(w2vp)+"\n")

print(str(len(globalConceptList)))
print(str(len(mmWordList)))
# globalConceptList.append(list(mmWordList))
globalConceptList = globalConceptList + mmWordList
print(str(len(globalConceptList)))

globalConceptList2 = list(set(globalConceptList))
w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, globalConceptList2)
testDatafilePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData.txt"
w2vFPList = computeFP(testDatafilePath, globalConceptList, gtConceptList)
w2vr = computeRecall(w2vFrequencyList, denom)
w2vp = computePrecision(w2vFrequencyList, w2vFPList)
resultWriteFile.write("withMetaMap\t"+str(w2vr)+"\t"+str(w2vp)+"\n")
print(str(w2vr)+"\t"+str(w2vp)+"\n")

resultWriteFile.close()




