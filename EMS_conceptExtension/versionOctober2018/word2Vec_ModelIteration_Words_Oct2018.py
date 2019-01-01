import gensim, logging
import io
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from itertools import repeat
import string

def fileLength(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


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



def loadConceptList(filepath):
    conceptWList = list()
    conceptPList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        items = line.lower().strip().split(" ")
        for item in items:
            if item not in conceptWList:
                conceptWList.append(item)

        # if line.find(" ") is not -1:
        #     conceptPList.append(line)
        # else:
        #     conceptWList.append(line)
    rfile.close()
    return conceptWList, conceptPList

def loadStopWords():
    filePath = "/Users/sarahmasudpreum/Desktop/PythonTCode/general/stopwords.txt"
    rfile = open(filePath, "r")
    stopWords = list()
    for line in rfile:
        stopWords.append(line.strip())
    rfile.close()
    return stopWords

def loadGTList (filePath):
    gtWList = list()
    gtPList = list()
    rfile = open(filePath, "r")
    itemList = list()
##    table = str.maketrans({key: None for key in string.punctuation})
    for line in rfile:
        line = line.lower().strip()
        line = re.sub("[^a-zA-Z0-9 ]"," ", line)
        stops = loadStopWords()
        if line.find(" ") is not -1:
            gtPList.append(line)
        else:
            gtWList.append(line)

    rfile.close()
    gtPList = list(set(gtPList))
    gtWList = list(set(gtWList))

    return gtWList, gtPList

def computeFrequency (gtConceptList, frequencyList, wordList):
    wordFrequencyList = list(repeat(0,len(gtConceptList)))
    for item in wordList:
        word = item.strip().lower()
        for i in range (0,len(gtConceptList)):
                if word.find(gtConceptList[i]) is not -1: #and gtConceptList[i].find(word) is not -1: #
                    wordFrequencyList[i] = frequencyList[i]
                    break

    
    print("^^^^^^^Frequency "+str(sum(wordFrequencyList)))
    return wordFrequencyList

def computeFP (filePath, conceptList, gtConceptList):
    filteredConceptList = [x.strip() for x in conceptList if x not in gtConceptList]
    filteredConceptList = list(set(filteredConceptList))
    print("#######FPListLength "+str(len(filteredConceptList)))
    with open(filePath, 'r') as rfile:
        data = rfile.read()
    # rfile = open(filePath, "r")
    frequencyList = list(repeat(0,len(filteredConceptList)))

    for i in range (0,len(filteredConceptList)):
        # print(data+"\n"+str(filteredConceptList[i]))
        # if data.find(str(filteredConceptList[i])) is not -1:
            # print(data+"\n"+str(filteredConceptList[i]))
            # print("___________")
        # frequencyList[i] = data.count(str(filteredConceptList[i]))
        if data.find(str(filteredConceptList[i])) is not -1:
            frequencyList[i] += 1


    # for line in rfile:
    #     for i in range (0,len(filteredConceptList)):
    #         wordTokens = line.lower().strip().split(" ")
    #         for word in wordTokens:
    #             word = word.strip()
    #             if word.find(filteredConceptList[i]) is not -1 and filteredConceptList[i].find(word) is not -1:
    #                 frequencyList[i] += 1

    rfile.close()
    # print("^^^^^^^FPcount "+str(sum(frequencyList)))
    return frequencyList

def computePrecision (TPlist, FPList):
    t = sum(TPlist) + sum(FPList)
    p = float(0)
    if t>0:
        p = sum(TPlist)/float( t )
    return round( p,4 )

def computeRecall (TPlist, denom):
    p = float(0)
    if denom>0:
        p = sum(TPlist)/float(denom)
    return round( p,4 )

##load groundTruth
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/FormattedTags/allInOne_new.txt"
##gtConceptList = loadSimpleList(filePath, True)
gtWList, gtPList = loadGTList(filePath)
gtConceptList = gtWList
frequencyList = list(repeat(0,len(gtConceptList)))

##load originalConceptList
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/conceptList.txt"
conceptWList, conceptPList = loadConceptList(filePath)
stops = loadStopWords()

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/MetaMapOutputsConciseW.txt" ## MetaMapOutputsConciseW MetaMapOutputs_FilteredST_ConciseW
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
        wordTokens = line.lower().strip().split(" ")
        for word in wordTokens:
            if word.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(word) is not -1:
                frequencyList[i] += 1            
rfile.close()
##create Output file
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/W2VTuning/Results_W2V_SG_MM_words_1e-3_10.txt" #DomainKnowledge_MM_
wfile = open (filePath, "w")
##wfile.write("cbow_mean\tvectorSize\twindowLength\ttopN\thfListSize\tRecall\tPrecision\n")
wfile.write("cbow_mean\tvectorSize\twindowLength\ttopN\tRecall\tPrecision\tRecall_MM\tPrecision_MM\n")

denom = sum(frequencyList)

##createModel
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/wordEmbeddingModel/Data/EMSallDataIn1.txt" 
n= fileLength(filePath)
sentenceList = loadTrainingData(filePath, n, True)

conceptList = conceptWList
##iterate parameters
min_count = 10
vectorSize = 50
windowSize = 5
cbow_mean = 0
downsampling = 1e-3
stops = loadStopWords()##set(stopwords.words("english"))

# for cbow_mean in [0, 1]:#[1,0]:
for vectorSize in [50, 100, 150, 200, 250, 300]:#[50, 100, 150, 200, 250, 300]:
    for windowSize in [5, 10, 15, 20]: #, 15, 20
        for TOPN in [25, 50, 100]:#[100, 150, 200]:
##                for hfListSize in [hfListSize-100, hfListSize]:
            model = gensim.models.Word2Vec(sentenceList, sg=1, min_count=min_count, size=vectorSize, window=windowSize, cbow_mean = cbow_mean,sample = downsampling)
##                model = gensim.models.Word2Vec(sentenceList, min_count=min_count, size=vectorSize, window=windowSize, sample = downsampling)

            ##createConcepts from model
            w2vList = set()
            wfile.write(str(cbow_mean)+"\t"+str(vectorSize)+"\t"+str(windowSize)+"\t"+str(TOPN)+"\t")
            filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/W2VTuning/Results_W2V_SG_MM_words_WordList_"+str(TOPN)+".txt" #DomainKnowledge_MM_
            wfile2 = open (filePath, "w")
            wfile2.write(str(vectorSize)+"\t"+str(windowSize)+"\t"+str(TOPN)+"\n")

##            print(str(vectorSize)+"\t"+str(windowSize)+"\t"+str(TOPN)+"\t")

            for concept in conceptList:
                candidateList = list()
                if concept in model:
                    wfile2.write(str(concept)+"\t")
                    candidateList.append((concept,1))
                    candidateList2 = model.most_similar(concept, topn = TOPN)
                    for item in candidateList2:
                        candidateList.append(item)
                
                for item in candidateList:
                    wfile2.write(str(item[0])+"\t")
                    if not item[0].isdigit():
                        w2vList.update(item[0])
                wfile2.write("\n")
            wfile2.close()
                        
                    
##                for item in mmWordList:
##                    if item not in w2vList:
##                        w2vList.append(item)
##
##   ##evaluateModel
##                for item in conceptList:
##                    if item not in w2vList:
##                        w2vList.append(item)
            w2vList.update(conceptList)
            w2vList = list(w2vList)
            w2vList = [w for w in w2vList if not w in stops and not w.isdigit()]

            w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, w2vList)
            filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
            w2vFPList = computeFP(filePath, w2vList, gtConceptList)

            w2vr = computeRecall(w2vFrequencyList, denom)
            w2vp = computePrecision(w2vFrequencyList, w2vFPList)
            wfile.write(str(w2vr)+"\t"+str(w2vp)+"\t")
            print(str(w2vr)+"\t"+str(w2vp)+"\t")
##add MM
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

####add MM+DK
##                w2vList = set(w2vList)
##                w2vList.update(conceptList)                
##                w2vList = list(w2vList)
##
##                w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, w2vList)
##                filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
##                w2vFPList = computeFP(filePath, w2vList, gtConceptList)
##
##                w2vr = computeRecall(w2vFrequencyList, denom)
##                w2vp = computePrecision(w2vFrequencyList, w2vFPList)
##                wfile.write(str(w2vr)+"\t"+str(w2vp)+"\n")
##                print(str(w2vr)+"\t"+str(w2vp)+"\n")

                
wfile.close()
