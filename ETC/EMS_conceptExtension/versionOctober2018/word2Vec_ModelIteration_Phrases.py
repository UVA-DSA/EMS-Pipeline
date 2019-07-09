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

def hasStopwords(inputString):
    stops = loadStopWords()
    tokens = inputString.split(" ")
    for item in stops:
        if item in tokens:
            return True
    return False

def fileLength(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def loadTrainingData(filepath, n, remove_stopwords=True):
    dataList = [[] for x in range(n)]
    rfile = open(filepath, "r")
    for line in rfile:
        line = line.lower().strip()
        line = re.sub("[^a-zA-Z0-9'-]"," ", line)
        stops = loadStopWords()
        innerList = line.split()
        innerList = [w for w in innerList if not w in stops and not w.isdigit()]
        dataList.append(innerList)
    rfile.close()
    return dataList


# def loadData(filepath, remove_stopwords=False):
# ##    dataList = list()
#     dataList = [[] for x in range(11891)]
#     rfile = open(filepath, "r")
#     for line in rfile:
#         line2 = line.lower().strip()
#         line2 = re.sub("[^a-zA-Z0-9_]"," ", line2)
#         innerList = line2.split()
#         if remove_stopwords:
#             stops = loadStopWords()
#             innerList = [w for w in innerList if not w in stops]
#         innerList = [w for w in innerList if not w.isdigit()]        
#         dataList.append(innerList)
#     rfile.close()
#     return dataList


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
        stopWords.append(line.lower().strip())
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


def computeFrequency (gtConceptList, frequencyList, wordList):
    wordFrequencyList = list(repeat(0,len(gtConceptList)))
    for item in wordList:
        for i in range (0,len(gtConceptList)):
            gTerm = str(gtConceptList[i]).lower().strip()
            item = item.lower().strip()
##            if len(gTerm.split(" ")) > 1:
            if item.find(gTerm) is not -1: #or gTerm.find(item) is not -1:
                wordFrequencyList[i] = frequencyList[i]

    # filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_phrases_AvgDist_TP_Phrases.txt"
    # wfile = open(filePath, "w")
    # # wfile.write("======================================================================================================\n")
    # finalList = list(zip(gtConceptList, wordFrequencyList))
    # for item in finalList:
    #     if item[1] > 0:
    #         wfile.write(str(item[0]) + " " + str(item[1])+ "\n")
    # wfile.close()
    return wordFrequencyList

def computeFP (filePath, conceptList, gtConceptList):
##    filteredConceptList = [x.strip() for x in conceptList if x not in gtConceptList]
    tpConceptList = list()
    for item in conceptList:
        for i in range (0,len(gtConceptList)):
            gTerm = str(gtConceptList[i]).lower().strip()
            item = item.lower().strip()
##            if len(gTerm.split(" ")) > 1:
            if item.find(gTerm) is not -1: #or gTerm.find(item) is not -1:
                tpConceptList.append(item)
    filteredConceptList = [w for w in conceptList if w not in tpConceptList]
    # filteredConceptList = [w for w in filteredConceptList if not hasNumbers(w)]
    # filteredConceptList = [w for w in filteredConceptList if not hasStopwords(w)]
    # filteredConceptList = [w for w in filteredConceptList if w.find("patient") is -1 or w.find(" pt") is -1] 

    filteredConceptList = list(set(filteredConceptList))
    rfile = open(filePath, "r")
    frequencyList = list(repeat(0,len(filteredConceptList)))
    stops = loadStopWords()##set(stopwords.words("english"))

    for line in rfile:
    	# tokens = word_tokenize(line) #word_tokenize
    	# tokens = [w for w in tokens if not w.isdigit()]
        line = re.sub("[^a-zA-Z0-9'-]"," ", line)
    	# tokens = line.split(" ")
    	# tokens = [w for w in tokens if w.find(" ") is -1 and not w.isdigit()]
    	# line = " ".join(tokens)
        line = line.lower().strip()
        for i in range (0,len(filteredConceptList)):
            if line.find(filteredConceptList[i]) is not -1:
                frequencyList[i] = 1

    rfile.close()
    # filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_phrases_AvgDist_FP_Phrases.txt"
    # wfile = open(filePath, "w")
    # # wfile.write("======================================================================================================\n")
    # finalList = list(zip(filteredConceptList, frequencyList))
    # for item in finalList:
    #     if item[1] > 0:
    #         wfile.write(str(item[0]) + " " + str(item[1])+ "\n")
    # wfile.close()
    return frequencyList

def computePrecision (TPlist, FPList):
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
	return sim_avg_score(seq1_word_list, seq2_word_list,model)

def sim_max_score(wordlist1, wordlist2,model):
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

		
def model_most_similar_phrases(model, inputSpace, seedConcepts, topn, vs, ws):
    candidateSet = set()
    filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_AvgDist_PhraseList_"+str(vs)+"_"+str(ws)+"_"+str(topn)+".txt" #DomainKnowledge_MM_
    wfile2 = open (filePath, "w")
    for sc in seedConcepts:
        wfile2.write(str(sc)+"\t")
        scoreList = list(repeat(0,len(inputSpace)))
        for i in range(0,len(inputSpace)):
            ip = inputSpace[i]
            scoreList[i] = score(ip, sc, model)
        finalList = list(zip(inputSpace, scoreList))
        finalList.append((sc,float(1.0)))
        finalList.sort(key=lambda x: x[1])
        finalList = finalList[-topn:]
        print(finalList)
        for item in finalList:
            if not hasNumbers(item[0]) and not hasStopwords(item[0]) and len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(item[0])) == 0:
                if item[0].find("patient") is -1 and item[0].find(" pt") is -1 and item[0].find(" t ") is -1 and item[0].find(" s ") is -1:
                    wfile2.write(str(item)+"\t")
        candidateSet.update(finalList)
        wfile2.write("\n")

    wfile2.close()
    temp = np.array(list(candidateSet))
    return list(temp[:,0] )

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

##create N-Grams
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/trainingData/DataLines.txt" 
dataString = loadStringData2(filePath)
dataString = re.sub("[^a-zA-Z0-9'-]"," ", dataString)
##        tokens = line.split(" ")
##        tokens = [w for w in tokens if w.find(" ") is -1 and not w.isdigit() and not w in stops]
##        line = " ".join(tokens)
# sent_tokenize_list = sent_tokenize(dataString)
# tokens = nltk.wordpunct_tokenize(dataString) #word_tokenize
tokens = word_tokenize(dataString) #word_tokenize
tokens = [w for w in tokens if not w.isdigit()]
stops = loadStopWords()

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/conceptList.txt"
conceptList = loadOriginalConceptList(filePath)
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

##filtTrigramList = set()
##for trigram in trigramList:
##    if not trigram[0] in stops and not trigram[2] in stops:
##        print(trigram[0]+" "+trigram[1]+" "+trigram[2] )
##        if trigram[0] in conceptList or trigram[1] in conceptList or trigram[2] in conceptList:
##            filtTrigramList.update(trigram[0]+" "+trigram[1]+" "+trigram[2])
##            
##len(filtTrigramList)




##load GT annotation
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/FormattedTags/allInOne_new.txt"
gtConceptList = loadSimpleListPhrase(filePath)
frequencyList = list(repeat(0,len(gtConceptList)))
##load protocol concept list
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/conceptList.txt"
conceptList = loadOriginalConceptListPhrase(filePath)
##load metaMap extended concept list
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/MetaMapOutputsConciseP.txt" ## MetaMapOutputsConciseW MetaMapOutputs_FilteredST_ConciseW
rfile1 = open(filePath,"r")
mmWordList = list()
for line in rfile1:
    word = line.strip()
    if not word.isdigit():
        if word not in mmWordList:
            mmWordList.append(word)
rfile1.close()
##count ground truth in testData
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
rfile = open(filePath, "r")
for line in rfile:
    for i in range (0,len(gtConceptList)):
        line = line.lower().strip()
        line = re.sub("[^a-zA-Z0-9'-]"," ", line)
        if line.find(gtConceptList[i]) is not -1:
            frequencyList[i] += 1            
rfile.close()
##create Output file
denom = sum(frequencyList)

###calculate phrase similarity
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/trainingData/DataLines.txt" 
n= fileLength(filePath)
sentenceList = loadTrainingData(filePath, n, True)
min_count = 1
vectorSize = 100
windowSize = 10
cbow_mean = 0
downsampling = 1e-3

# model = gensim.models.Word2Vec(sentenceList, sg=0, min_count=min_count, size=vectorSize, window=windowSize, sample = downsampling)

candidateList = list()
inputSpace = set()
inputSpace.update(filtTrigramList)
inputSpace.update(filtBigramList)
inputSpace = list(inputSpace)
# inputSpace = inputSpace[5500:5505]
# conceptList = conceptList[1:10]
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_phrases_AvgDist.txt" #DomainKnowledge_MM_
wfile = open (filePath, "a")
wfile.write("cbow_mean\tvectorSize\twindowLength\ttopN\tRecall\tPrecision\tRecallMM\tPrecisionMM\n")
##wfile.write("cbow_mean\tvectorSize\twindowLength\ttopN\tRecall\tPrecision\n")


################################################################################################################################################
##candidateList = model_most_similar_phrases(model, inputSpace, conceptList, topn=200)
##w2vList.update(candidateList)
##w2vList = list(w2vList)
##for item in w2vList:
####    print(item)
##    wfile.write(item+"\n")
##wfile.close()
################################################################################################################################################

for cbow_mean in [0]:#[1,0]:
    for vectorSize in [100]:#[50, 100, 150, 200, 250, 300]:
        for windowSize in [10]: #, 15, 20
            for TOPN in [100]:#[100, 150, 200]:
                model = gensim.models.Word2Vec(sentenceList, sg=0, min_count=min_count, size=vectorSize, window=windowSize, cbow_mean = cbow_mean,sample = downsampling)
                ##createConcepts from model
                candidateList = list()
                w2vList = set()
                wfile.write(str(cbow_mean)+"\t"+str(vectorSize)+"\t"+str(windowSize)+"\t"+str(TOPN)+"\t")
                candidateList = model_most_similar_phrases(model, inputSpace, conceptList, TOPN, vectorSize, windowSize)
                w2vList.update(candidateList)
                w2vList = list(w2vList)             
                w2vList = [w for w in w2vList if not hasNumbers(w)]
                w2vList = [w for w in w2vList if not hasStopwords(w)]
                w2vList = [w for w in w2vList if w.find("patient") is -1]
                w2vList = [w for w in w2vList if w.find(" pt") is -1]
                w2vList = [w for w in w2vList if w.find(" t ") is -1]
                w2vList = [w for w in w2vList if w.find(" s ") is -1]

                # and w.find(" pt") is -1 and w.find(" t ") is -1 and w.find(" s ") is -1] 
                w2vList= [w for w in w2vList if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(w)) == 0 ]
                w2vList = set (w2vList)
                # w2vList.update(mmWordList)
                w2vList.update(conceptList)
                # w2vList= [w for w in w2vList if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(w)) == 0 ]
                w2vList = list(w2vList)              
                # w2vList = [w for w in w2vList if w.find("patient") is -1 or w.find(" pt") is -1 or w.find(" t ") is -1 or w.find(" s ") is -1] 
                # w2vList= [w for w in w2vList if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(w)) == 0 ]


                filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_phrases_AvgDist_modelOutputPhrases"+str(vectorSize)+"_"+str(windowSize)+"_"+str(TOPN)+".txt"
                wfile2 = open (filePath, "w")
                wfile2.write("======================================================================================================\n")
                for item in w2vList:
                	if len(set('[~!@#$%^&*()_-+{}<>":,.;?/|\']+$').intersection(item)) == 0 and item.find("patient") is -1:
	                    wfile2.write(item+"\n")
                wfile2.close()
          
##              EvaluateModel
                w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, w2vList)
                filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
                w2vFPList = computeFP(filePath, w2vList, gtConceptList)

                w2vr = computeRecall(w2vFrequencyList, denom)
                w2vp = computePrecision(w2vFrequencyList, w2vFPList)
                wfile.write(str(w2vr)+"\t"+str(w2vp)+"\t")
                print(str(w2vr)+"\t"+str(w2vp)+"\n")

##              EvaluateModel MetaMap
                w2vList = set(w2vList)
                w2vList.update(mmWordList)
                w2vList = list(w2vList)

                w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, w2vList)
                filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
                w2vFPList = computeFP(filePath, w2vList, gtConceptList)

                w2vr = computeRecall(w2vFrequencyList, denom)
                w2vp = computePrecision(w2vFrequencyList, w2vFPList)
                wfile.write(str(w2vr)+"\t"+str(w2vp)+"\n")
                print(str(w2vr)+"\t"+str(w2vp)+"\n")


                
wfile.close()
