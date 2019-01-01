from itertools import repeat
import string
from nltk.corpus import stopwords
import re

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



def loadConceptList2(filepath):
    conceptWList = list()
    conceptPList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        line = line.lower().strip()
        if line.find(" ") is not -1:
            if line.find("\t") is -1:
                conceptPList.append(line)
            else:
                tokens = line.split("\t")
                for token in tokens:
                    conceptPList.append(line)
        else:
            conceptWList.append(line)
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
        line = re.sub("[^a-zA-Z0-9'-]"," ", line)
        stops = loadStopWords()
        if line.find(" ") is not -1:
            gtPList.append(line)
        else:
            gtWList.append(line)

    rfile.close()
    gtPList = list(set(gtPList))
    gtWList = list(set(gtWList))

    return gtWList, gtPList
    

##def loadMMConceptList(filePath):
##    conceptWList = list()
##    conceptPList = list()
##    rfile = open(filePath, "r")    
##    for line in rfile:
##        line = re.sub("[^a-zA-Z0-9_]"," ", line)
##        if line.find(" ") is not -1:
##            conceptPList.append(line)
##        else:
##            conceptWList.append(line)
####        items = line.lower().strip().split(" ")
####        for item in items:
####            item = str(item)
####            if not item.isdigit():
####                if item not in conceptList:
####                    conceptList.append(item)
####    stops = loadStopWords    
##    rfile.close()
##    return conceptWList, conceptPList


def loadMMConceptList(filePath):
    conceptList = list()
    rfile = open(filePath, "r")    
    for line in rfile:
        line = line.lower().strip()
##        line = re.sub("[^a-zA-Z0-9'-]"," ", line)
        if line not in conceptList:
            conceptList.append(line)
##        items = line.lower().strip().split(" ")
##        for item in items:
##            item = str(item)
##            if not item.isdigit():
##                if item not in conceptList:
##                    conceptList.append(item)
    stops = loadStopWords()
    conceptWList = [w for w in conceptList if w.find(" ") is -1]
    conceptList = [w for w in conceptList if not w in conceptWList]
    rfile.close()
    return conceptWList, conceptList

def loadConceptList(filePath, fielName):
    conceptList = list()
##    if "w2v" in fielName:
##        filePath = filePath + "w2v" + ".txt"
##    if "d2v" in fielName:
##        filePath = filePath + "d2v" + ".txt"
    if "i2b2" in fielName:
        filePath = filePath + "i2b2" + ".txt"
    if "google" in fielName:
        filePath = filePath + "google" + ".txt"
    rfile = open(filePath, "r")
    
    conceptWList = list()
    conceptPList = list()
    rfile = open(filePath, "r")
    for line in rfile:
        line = re.sub("[^a-zA-Z0-9'-]"," ", line)
        line = line.lower().strip()
        if line.find(" ") is not -1:
            if line.find("\t") is -1:
                conceptPList.append(line)
            else:
                tokens = line.split("\t")
                for token in tokens:
                    conceptPList.append(token)
        else:
            conceptWList.append(line)
    rfile.close()
    return conceptWList, conceptPList




def computeFrequency (gtConceptList, frequencyList, wordList):
    wordFrequencyList = list(repeat(0,len(gtConceptList)))
    for item in wordList:
        for i in range (0,len(gtConceptList)):
            wordTokens = item.lower().strip().split(" ")
            for word in wordTokens:
                if word.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(word) is not -1:
                    wordFrequencyList[i] = frequencyList[i]

##    filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_MM_words_TPWordList.txt"
##    wfile = open(filePath, "a")
##    wfile.write("======================================================================================================\n")
##    finalList = list(zip(gtConceptList, wordFrequencyList))
##    for item in finalList:
##        if item[1] > 0:
##            wfile.write(str(item[0]) + " " + str(item[1])+ "\n")
##    wfile.close()
    return wordFrequencyList

def computeFP (filePath, conceptList, gtConceptList):
    filteredConceptList = [x.strip() for x in conceptList if x not in gtConceptList]
    filteredConceptList = list(set(filteredConceptList))
    rfile = open(filePath, "r")
    frequencyList = list(repeat(0,len(filteredConceptList)))

    for line in rfile:
        for i in range (0,len(filteredConceptList)):
            wordTokens = line.lower().strip().split(" ")
            for word in wordTokens:
                word = word.strip()
                if word.find(filteredConceptList[i]) is not -1 and filteredConceptList[i].find(word) is not -1:
                    frequencyList[i] += 1

    rfile.close()
##    filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/May20/RecallPrecision/Results_W2V_CBOW_MM_words__FPWordList.txt"
##    wfile = open(filePath, "a")
##    wfile.write("======================================================================================================\n")
##    finalList = list(zip(filteredConceptList, frequencyList))
##    for item in finalList:
##        if item[1] > 0:
##            wfile.write(str(item[0]) + " " + str(item[1])+ "\n")
##    wfile.close()
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

##load groundTruth
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/FormattedTags/allInOne_new.txt"
##gtConceptList = loadSimpleList(filePath, True)
gtWList, gtPList = loadGTList(filePath)
gtConceptList = gtWList
frequencyList = list(repeat(0,len(gtConceptList)))
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
##finalList = list(zip(gtConceptList, frequencyList))
##for item in finalList:
##    if item[1] > 0:
##        print(str(item[0]) + " " + str(item[1]))

##count word frequnecy in models
##load originalConceptList
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/conceptList.txt"
conceptWList, conceptPList = loadConceptList2(filePath)
stops = loadStopWords()


##w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, w2vList)
##d2vFrequencyList = computeFrequency (gtConceptList, frequencyList, d2vList)
SpcFrequencyList = computeFrequency (gtConceptList, frequencyList, conceptWList)

denom = sum(frequencyList)
##finalList = list(zip(gtConceptList, frequencyList, w2vFrequencyList, d2vFrequencyList, i2b2FrequencyList, googleFrequencyList, mmFrequencyList))
##for item in finalList:
##    if item[1] > 0:
##        print(str(item[0]) + " " + str(item[2]) + " " + str(item[3]) + " " + str(item[4]) + " " + str(item[5]) + " " + str(item[6]))


filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
##w2vFPList = computeFP(filePath, w2vList, gtConceptList)
##d2vFPList = computeFP(filePath, d2vList, gtConceptList)
SpcFPList = computeFP(filePath, conceptWList, gtConceptList)
print("Recall\t" + str(round(sum(SpcFrequencyList)/float(denom),4)))

##print(sum(frequencyList), sum(w2vFrequencyList), sum(d2vFrequencyList), sum(i2b2FrequencyList), sum(googleFrequencyList), sum(mmFrequencyList), sum(mmFilteredFrequencyList))
##print(sum(w2vFPList), sum(d2vFPList), sum(i2b2FPList), sum(googleFPList), sum(mmFPList), sum(mmFilteredFPList))
##w2vp = computePrecision(w2vFrequencyList, w2vFPList)
##d2vp = computePrecision(d2vFrequencyList, d2vFPList)
Spcp = computePrecision(SpcFrequencyList, SpcFPList)

##wfile.write("ModelName\tWord2Vec\tDoc2Vec\tI2B2\tGoogle\tMMoutput\tMMoutputFiltST\tMM_ManualList\tMMList\tMMListFiltST\n")
##wfile.write("Precision\t"+str(w2vp)+ "\t" + str(d2vp)+ "\t" + str(i2b2p)+ "\t" + str(gp)+ "\t" + str(mmp)+ "\t" + str(mm_FiltST_p)+ "\t" + str(mmManp) + "\t"+ str(mmListp)+ "\t" + str(mm_FiltST_Listp)+ "\n")
print("Precision\t"+ str(Spcp))

