from itertools import repeat
import string
from nltk.corpus import stopwords
import re

def isNumber(inputString):
    try:
        float(inputString)
        return True
    except ValueError:
        return False

def loadMMConceptList(filePath):
    conceptList = list()
    rfile = open(filePath, "r")    
    for line in rfile:
        line = re.sub("[^a-zA-Z0-9_]"," ", line)
        items = line.lower().strip().split(" ")
        for item in items:
            item = str(item)
            if not item.isdigit():
                if item not in conceptList:
                    conceptList.append(item)
    rfile.close()
    return conceptList


    
def loadSimpleList (filePath, splitFlag = False):
    rfile = open(filePath, "r")
    itemList = list()
    table = str.maketrans({key: None for key in string.punctuation})
    for line in rfile:
        item = line.lower().strip()
        if splitFlag:
            items = item.split(" ")
            stops = set(stopwords.words("english"))
            items = [w for w in items if not w in stops]
            for i in items:
                i = i.translate(table)
                if i not in itemList:
                    itemList.append(i)
        else:
            if item not in itemList:
                itemList.append(item)
    rfile.close()
    return itemList

def computeFrequency (gtConceptList, frequencyList, wordList):
    wordFrequencyList = list(repeat(0,len(gtConceptList)))
    for item in wordList:
        for i in range (0,len(gtConceptList)):
            wordTokens = item.lower().strip().split(" ")
            for word in wordTokens:
                if word.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(word) is not -1:
                    wordFrequencyList[i] = frequencyList[i]
    return wordFrequencyList

def computeFP (filePath, conceptList, gtConceptList):
    filteredConceptList = [x for x in conceptList if x not in gtConceptList]
    stops = set(stopwords.words("english"))
    filteredConceptList = [w for w in filteredConceptList if not w in stops]
    rfile = open(filePath, "r")
    frequencyList = list(repeat(0,len(filteredConceptList)))
    for line in rfile:
        line = re.sub("[^a-zA-Z0-9_]"," ", line)
        for i in range (0,len(filteredConceptList)):
            concept = str(filteredConceptList[i])
##            print(concept)
            wordTokens = line.lower().strip().split(" ")
            for word in wordTokens:
##                if not concept.isdigit():
                    if word.find(concept) is not -1 and concept.find(word) is not -1:
                        frequencyList[i] += 1
    print(str(len(frequencyList)))
    print(str(sum(frequencyList)))
    return frequencyList

def computePrecision (TPlist, FPList):
    p = sum(TPlist)/float( sum(TPlist) + sum(FPList))
    return round( p,4 )

def computeRecall (TPlist, denom):
    p = sum(TPlist)/float(denom)
    return round( p,4 )

#################################################################################
##load groundTruth
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/FormattedTags/allInOne.txt"
gtConceptList = loadSimpleList(filePath, True)
frequencyList = list(repeat(0,len(gtConceptList)))

##count ground truth in testData
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData.txt"
rfile = open(filePath, "r")
for line in rfile:
    for i in range (0,len(gtConceptList)):
##        if len(gtConceptList[i]) > 2:
##            if line.find(gtConceptList[i]) is not -1:
##                frequencyList[i] += 1
##        else:
        wordTokens = line.lower().strip().split(" ")
        for word in wordTokens:
            if word.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(word) is not -1:
                frequencyList[i] += 1            
rfile.close()

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/MetaMapOutputsConciseW.txt"
mmList = loadMMConceptList(filePath)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/MetaMapOutputs_FilteredST_ConciseW.txt"
mm_FiltST_List = loadMMConceptList(filePath)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/MetaMapExtendedConceptList_JavaAPI.txt"
mmExtConceptList = loadMMConceptList(filePath)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/MetaMapExtendedConceptList_FilteredST_JavaAPI.txt"
mm_FiltST_ExtConceptList = loadMMConceptList(filePath)

##construct TP list
mmFrequencyList = computeFrequency (gtConceptList, frequencyList, mmList)
mm_FiltST_FrequencyList = computeFrequency (gtConceptList, frequencyList, mm_FiltST_List)
mmExtConceptFrequencyList = computeFrequency (gtConceptList, frequencyList, mmExtConceptList)
mm_FiltST_ExtConceptFrequencyList = computeFrequency (gtConceptList, frequencyList, mm_FiltST_ExtConceptList)

denom = sum(frequencyList)
finalList = list(zip(gtConceptList, frequencyList, mmFrequencyList, mm_FiltST_FrequencyList, mmExtConceptFrequencyList, mm_FiltST_ExtConceptFrequencyList))
for item in finalList:
    if item[1] > 0:
        print(str(item[0]) + " " + str(item[1]) + " " + str(item[2])+" " +str(item[3]) + " " + str(item[4]) + " " + str(item[5]))


##finalList = list(zip(gtConceptList, frequencyList, w2vFrequencyList, d2vFrequencyList, i2b2FrequencyList, googleFrequencyList, mmFrequencyList))
##for item in finalList:
##    if item[1] > 0:
##        print(str(item[0]) + " " + str(item[2]) + " " + str(item[3]) + " " + str(item[4]) + " " + str(item[5]) + " " + str(item[6]))


filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData.txt"
mmFPList = computeFP(filePath, mmList, gtConceptList)
mm_FiltST_FPList = computeFP(filePath, mm_FiltST_List, gtConceptList)
mmExtConceptFPList = computeFP(filePath, mmExtConceptList, gtConceptList)
mm_FiltST_ExtConceptFPList = computeFP(filePath, mm_FiltST_ExtConceptList, gtConceptList)

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/version2/MM_Results.txt"
wfile = open (filePath, "w")
##wfile.write(str(sum(w2vFrequencyList))+ "\t" + str(sum(d2vFrequencyList))+ "\t" + str(sum(i2b2FrequencyList))+ "\t" + str(sum(googleFrequencyList))+ "\t" +  str(sum(mmFrequencyList)) +"\n")
##wfile.write(str(sum(w2vFPList))+ "\t" + str(sum(d2vFPList))+ "\t" + str(sum(i2b2FPList))+ "\t" + str(sum(googleFPList))+ "\t" +  str(sum(mmFPList)) +"\n")
wfile.write("ModelName\t MMoutput\tMMoutputFiltST\tMMList\tMeMListFiltST\n")
wfile.write("Recall\t"+str(round(sum(mmFrequencyList)/float(denom),4))+ "\t" + str(round(sum(mm_FiltST_FrequencyList)/float(denom),4))+ "\t" +str(round(sum(mmExtConceptFrequencyList)/float(denom),4))+ "\t" + str(round(sum(mm_FiltST_ExtConceptFrequencyList)/float(denom),4))+"\n")

mmp = computePrecision(mmFrequencyList, mmFPList)
mm_FiltST_p = computePrecision(mm_FiltST_FrequencyList, mm_FiltST_FPList)
mmListp = computePrecision(mmExtConceptFrequencyList, mmExtConceptFPList)
mm_FiltST_Listp = computePrecision(mm_FiltST_ExtConceptFrequencyList, mm_FiltST_ExtConceptFPList)


wfile.write("ModelName\t MMoutput\tMMoutputFiltST\tMMList\tMeMListFiltST\n")
wfile.write("Precision\t"+str(mmp)+ "\t" + str(mm_FiltST_p)+ "\t" +str(mmListp)+ "\t" + str(mm_FiltST_Listp)+ "\n")

wfile.close()
