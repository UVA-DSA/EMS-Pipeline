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

def hasRedundantToken(inputString):
    tokens = inputString.split(" ")
    for token in tokens:
        if token.find("patient") is not -1 or token.find("pt") is not -1 or len(token)==1:
            return True

    return False
def hasSpecialCharacter(inputString):
    # len(set('[~!@#$%^&*()_-+{}<>"=:,.;?/|\']+$').intersection(w)) == 0
    temp = re.sub("[^a-zA-Z0-9 ]","*", inputString)
    if len(set('*').intersection(temp)) > 0:
        return True
    return False
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
        item = line.strip()
        if item not in conceptList:
            conceptList.append(item)
    rfile.close()
    print(conceptList)
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
            if item.find(gTerm) is not -1: #or gTerm.find(item) is not -1:
                wordFrequencyList[i] = frequencyList[i]

    # filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/candidatePhrases/"
    # filePath = filePath + fileName +".txt"
    # filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/PhraseLevel/conceptPhrase/Results_W2V_CBOW_DomainKnowledge_MM_modFP_newStopWords_MaxAvgDist_TP_Phrases.txt"
    # wfile = open(filePath, "w")
    # wfile.write("======================================================================================================\n")
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
            gTerm = str(gtConceptList[i]).strip()
            item = item.strip()
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
    	# line = re.sub("[^a-zA-Z0-9]"," ", line)
    	# tokens = line.split(" ")
    	# tokens = [w for w in tokens if w.find(" ") is -1 and not w.isdigit()]
    	# line = " ".join(tokens)
    	for i in range (0,len(filteredConceptList)):
    		if line.find(filteredConceptList[i]) is not -1:
    			frequencyList[i] = 1

    rfile.close()
    # filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/PhraseLevel/conceptPhrase/Results_W2V_CBOW_DomainKnowledge_MM_modFP_newStopWords_MaxAvgDist_FP_Phrases.txt"
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
    finalList.sort(key=lambda x: x[1], reverse=True)
    finalList = finalList[:topn]
    return finalList
        # finalList = finalList[-topn:]
        # # print(finalList)
        # candidateSet.update(finalList)
            
        # temp = np.array(list(candidateSet))
        # candidateList = list(temp[:,0] )
        # scoreTupleList = list(candidateSet)
    # print("*******\n*******\n")
    # print(temp)
    # return list(temp[:,0] )
        # return candidateList, scoreTupleList

def createFileName(modelName, n=25):
    filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/ODEMSA/Phrase/"
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
dataFileName = "EMSallDataIn1.txt" #EMSallDataIn1 MedallDataIn1 MedEMSallDataIn1
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

        # filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/newConceptList_November2018.txt" # newConceptList_Sept2018 conceptList
        # conceptList = loadOriginalConceptList(filePath)
        # conceptList = [w for w in conceptList if w.find(" ") is -1]

finder = BigramCollocationFinder.from_words(tokens)
bigram_measures = nltk.collocations.BigramAssocMeasures()
##sorted(finder.nbest(bigram_measures.raw_freq, 100))
scored = finder.score_ngrams(bigram_measures.raw_freq)
bigramSet = set(sorted(bigram for bigram, score in scored))  
bigramList = list(bigramSet)
len(bigramList)
filtBigramList = list()
# filtBigramList= [u+" "+v for u,v in bigramList if not u in stops and not v in stops and (u in conceptList or v in conceptList)]
filtBigramList= [u+" "+v for u,v in bigramList if not u in stops and not v in stops]
len(filtBigramList)
filtBigramList= [w for w in filtBigramList if not hasSpecialCharacter(w) and not hasRedundantToken(w)]


# filtBigramList= [w for w in filtBigramList if len(set('[~!@#$%^&*()_-+\—{}<>"=\…\“\’:,.;?/|\']+$').intersection(w)) == 0 ]
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
# filtTrigramList= [u+" "+v+" "+w for u,v,w in trigramList if not u in stops and not v in stops and not w in stops and (u in conceptList or v in conceptList or w in conceptList)]
filtTrigramList= [u+" "+v+" "+w for u,v,w in trigramList if not u in stops and not v in stops and not w in stops]
len(filtTrigramList)
filtTrigramList= [w for w in filtTrigramList if not hasSpecialCharacter(w) and not hasRedundantToken(w)]

# filtTrigramList= [w for w in filtTrigramList if len(set('[~!@#$%^&*()_-+{}<>"=:,.;?/|\']+$').intersection(w)) == 0 ]
# filtTrigramList = [w for w in filtTrigramList if w.find("patient") is -1 and w.find(" pt") is -1] 
##filtTrigramList= [u+" "+v+" "+w for u,v,w in trigramList if not u in stops and not w in stops and (u in conceptList or v in conceptList or w in conceptList)]
filtTrigramList = list(set(filtTrigramList))
len(filtTrigramList)



##load GT annotation

##load protocol concept list
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/newConceptList_November2018.txt" # newConceptList_Sept2018 conceptList
conceptList = loadOriginalConceptListPhrase(filePath)
print(len(conceptList))
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
inputSpace = [w for w in inputSpace if not hasNumbers(w)]
inputSpace = [w for w in inputSpace if not hasStopwords(w)]
inputSpace= [w for w in inputSpace if not hasSpecialCharacter(w) and not hasRedundantToken(w)]

# model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sarahmasudpreum/src/word2vecModels/GoogleNews-vectors-negative300.bin', binary=True)  
# word2vec_RAA_Data_150_10 word2vec_Forum_Data_150_10 word2vec_MIMIC3_Data_150_10 word2vec_I2B2_Data_150_10
# word2vec_Med_Data_150_10 word2vec_EMS_Data_150_10 word2vec_MedEMS_Data_150_10
model = gensim.models.Word2Vec.load('/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/wordEmbeddingModel/Model/word2vec_EMS_Data_150_10.model')  
vl = model.wv.vocab ##model.vocab (for google) model.wv.vocab
topnDef = 50
f1, f2 = createFileName("EMS", topnDef) 
## RAA Forum MIMIC3 I2B2 Med EMS all Google Glove
wfile = open(f1, "w")
wfile2 = open(f2, "w")
# candidateList = list()
TOPN = 100

for concept in conceptList:
    # candidateList = list()
    # if concept in vl:
    # print(concept)
    # candidateList, scoreTupleList = model_most_similar_phrases(model, inputSpace, concept, topn=topnDef)
    # candidateList = [w for w in candidateList if not hasNumbers(w)]
    # candidateList = [w for w in candidateList if not hasStopwords(w)]
    # # candidateList = [w for w in candidateList if w.find("patient") is -1 or w.find(" pt") is -1 or w.find(" t ") is -1 or w.find(" s ") is -1] 
    # candidateList= [w for w in candidateList if not hasSpecialCharacter(w) and not hasRedundantToken(w)]
    # candidateList = list(set(candidateList))
    # if concept.find(" ") is not -1:
    #     print(concept, str(candidateList))
    scoreTupleList = model_most_similar_phrases(model, inputSpace, concept, topn=topnDef)
    print(concept)
    if len(scoreTupleList) >= topnDef:
        wfile.write(concept+"\t")
        wfile2.write(concept+"\t")
        # for item in candidateList:
        #     if not isNumber(item) and item.find("numberToken") is -1: 
        #         wfile.write(str(item)+"\t")
        #         for conceptScore in scoreTupleList:
        #             if conceptScore[0].find(item) is not -1 and item.find(conceptScore[0]) is not -1: 
        #                 wfile2.write(str(item)+","+str(conceptScore[1])+"\t")
        #                 # print(str(item)+","+str(conceptScore[1])+"\t")
        #                 break
        scoreTupleList.sort(key=lambda x: x[1], reverse= True)
        for item in scoreTupleList:
            if not isNumber(item[0]) and item[0].find("numberToken") is -1: 
                wfile.write(str(item[0])+"\t")
                wfile2.write(str(item[0])+","+str(item[1])+"\t")


        wfile.write("\n")
        wfile2.write("\n")

wfile.close()
wfile2.close()


