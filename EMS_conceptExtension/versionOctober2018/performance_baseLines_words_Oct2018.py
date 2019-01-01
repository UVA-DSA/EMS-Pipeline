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


# def loadTrainingData(filepath, n, remove_stopwords=True):
#     dataList = [[] for x in range(n)]
#     rfile = open(filepath, "r")
#     for line in rfile:
#         line = line.lower().strip()
#         line = re.sub("[^a-zA-Z0-9'-]"," ", line)
#         stops = loadStopWords()
#         innerList = line.split()
#         innerList = [w for w in innerList if not w in stops and not w.isdigit()]
#         dataList.append(innerList)
#     rfile.close()
#     return dataList



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
                    conceptPList.append(token)
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

def loadConceptList(filePath):
    conceptWList = list()
    rfile = open(filePath, "r")
    for line in rfile:
        # line = re.sub("[^a-zA-Z0-9'-]"," ", line)
        line = line.lower().strip()
        tokens = line.split("\t")
        for token in tokens:
            if token not in conceptWList:
                conceptWList.append(token)

    rfile.close()
    return conceptWList


# def computeFrequency (gtConceptList, frequencyList, wordList):
#     wordFrequencyList = list(repeat(0,len(gtConceptList)))
#     for item in wordList:
#         word = item.strip().lower()
#         for i in range (0,len(gtConceptList)):
#                 if word.find(gtConceptList[i]) is not -1: #and gtConceptList[i].find(word) is not -1: #
#                     wordFrequencyList[i] = frequencyList[i]
#                     break

    
#     # print("^^^^^^^Frequency "+str(sum(wordFrequencyList)))
#     return wordFrequencyList

# # def computeFrequency (gtConceptList, frequencyList, wordList):
# #     wordFrequencyList = list(repeat(0,len(gtConceptList)))
# #     for item in wordList:
# #         for i in range (0,len(gtConceptList)):
# #             wordTokens = item.lower().strip().split(" ")
# #             for word in wordTokens:
# #                 if word.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(word) is not -1:
# #                     wordFrequencyList[i] = frequencyList[i]

# #     return wordFrequencyList

# def computeFP (filePath, conceptList, gtConceptList):
#     filteredConceptList = [x.strip() for x in conceptList if x not in gtConceptList]
#     filteredConceptList = list(set(filteredConceptList))
#     # print("#######FPListLength "+str(len(filteredConceptList)))
#     with open(filePath, 'r') as rfile:
#         data = rfile.read()
#     # rfile = open(filePath, "r")
#     frequencyList = list(repeat(0,len(filteredConceptList)))

#     for i in range (0,len(filteredConceptList)):
#         # print(data+"\n"+str(filteredConceptList[i]))
#         # if data.find(str(filteredConceptList[i])) is not -1:
#             # print(data+"\n"+str(filteredConceptList[i]))
#             # print("___________")
#         # frequencyList[i] = data.count(str(filteredConceptList[i]))
#         if data.find(str(filteredConceptList[i])) is not -1:
#             frequencyList[i] += 1


#     # for line in rfile:
#     #     for i in range (0,len(filteredConceptList)):
#     #         wordTokens = line.lower().strip().split(" ")
#     #         for word in wordTokens:
#     #             word = word.strip()
#     #             if word.find(filteredConceptList[i]) is not -1 and filteredConceptList[i].find(word) is not -1:
#     #                 frequencyList[i] += 1

#     rfile.close()
#     # print("^^^^^^^FPcount "+str(sum(frequencyList)))
#     return frequencyList

def computeFrequency (gtConceptList, frequencyList, wordList):
    wordFrequencyList = list(repeat(0,len(gtConceptList)))
    for item in wordList:
        flag = 1
        for i in range (0,len(gtConceptList)):
            wordTokens = item.lower().strip()#.split(" ")
            if wordTokens.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(wordTokens) is not -1:
               wordFrequencyList[i] = frequencyList[i]
               flag = 0
               print(item)
               break
            # else:
            #     print("*****Mismatch: Ground Truth: "+gtConceptList[i]+"\t TestWord: "+item)
    for i in range (0,len(gtConceptList)):
        if wordFrequencyList[i] is 0:
            print("Mismatch:\t"+gtConceptList[i])

            # for word in wordTokens:
            #     if word.find(gtConceptList[i]) is not -1 and gtConceptList[i].find(word) is not -1:
            #         wordFrequencyList[i] = frequencyList[i]
            #         # break
            #     else:
            #         flag = 1
                    # temp = item
        # if flag is 1:
        #     print("Ground Truth: "+gtConceptList[i]+"\t TestWord: "+item)


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
            wordTokens = line.lower().strip()#.split(" ")
            # if wordTokens.find(filteredConceptList[i]) is not -1 and filteredConceptList[i].find(wordTokens) is not -1:
            #         frequencyList[i] += 1
            for word in wordTokens:
                word = word.strip()
                if word.find(filteredConceptList[i]) is not -1 and filteredConceptList[i].find(word) is not -1:
                    frequencyList[i] = 1

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
    # print("*** Precision "+str(t))
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

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "RAA"+".txt"
RAAList = loadConceptList(filePath)
# print("RAAListLen: "+str(len(RAAList)))
# for item in RAAList:
#     print(item)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "Forum"+".txt"
ForumList = loadConceptList(filePath)
# print("ForumListLen: "+str(len(ForumList)))

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "EMS"+".txt"
EMSList = loadConceptList(filePath)

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "I2B2"+".txt"
I2B2List = loadConceptList(filePath)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "MIMIC3"+".txt"
MIMIC3List = loadConceptList(filePath)
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "Med"+".txt"
MedList = loadConceptList(filePath)

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/25candidateConcepts_" + "MedEMS"+".txt"
MedEMSList = loadConceptList(filePath)
# print("MedEMSListLen: "+str(len(MedEMSList)))


# googleList, googlePList  = loadConceptList(filePath,"google")
# filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/metaMapExtendedConceptListFiltered.txt"


##w2vFrequencyList = computeFrequency (gtConceptList, frequencyList, w2vList)
##d2vFrequencyList = computeFrequency (gtConceptList, frequencyList, d2vList)
print("RAAFrequencyList")
# print("===================================================================")
RAAFrequencyList = computeFrequency (gtConceptList, frequencyList, RAAList)
# print("===================================================================")
# print("ForumFrequencyList")
ForumFrequencyList = computeFrequency (gtConceptList, frequencyList, ForumList)
print("===================================================================")
print("EMSFrequencyList")
EMSFrequencyList = computeFrequency (gtConceptList, frequencyList, EMSList)
# print("===================================================================")
# print("I2B2FrequencyList")
I2B2FrequencyList = computeFrequency (gtConceptList, frequencyList, I2B2List)
# print("===================================================================")
# print("MIMIC3FrequencyList")
MIMIC3FrequencyList = computeFrequency (gtConceptList, frequencyList, MIMIC3List)
print("===================================================================")
print("MedFrequencyList")
MedFrequencyList = computeFrequency (gtConceptList, frequencyList, MedList)
print("===================================================================")
print("MedEMSFrequencyList")
MedEMSFrequencyList = computeFrequency (gtConceptList, frequencyList, MedEMSList)
print("===================================================================")
# googleFrequencyList = computeFrequency (gtConceptList, frequencyList, googleList)


denom = sum(frequencyList)
# print("denom"+str(denom))
##finalList = list(zip(gtConceptList, frequencyList, w2vFrequencyList, d2vFrequencyList, i2b2FrequencyList, googleFrequencyList, mmFrequencyList))
##for item in finalList:
##    if item[1] > 0:
##        print(str(item[0]) + " " + str(item[2]) + " " + str(item[3]) + " " + str(item[4]) + " " + str(item[5]) + " " + str(item[6]))


filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/testData/testData_new.txt"
##w2vFPList = computeFP(filePath, w2vList, gtConceptList)
##d2vFPList = computeFP(filePath, d2vList, gtConceptList)
RAAFPList = computeFP(filePath, RAAList, gtConceptList)
ForumFPList = computeFP(filePath, ForumList, gtConceptList)
EMSFPList = computeFP(filePath, EMSList, gtConceptList)
I2B2FPList = computeFP(filePath, I2B2List, gtConceptList)
MIMIC3FPList = computeFP(filePath, MIMIC3List, gtConceptList)
MedFPList = computeFP(filePath, MedList, gtConceptList)
MedEMSFPList = computeFP(filePath, MedEMSList, gtConceptList)
# googleFPList = computeFP(filePath, googleList, gtConceptList)

##print(sum(frequencyList), sum(w2vFrequencyList), sum(d2vFrequencyList), sum(i2b2FrequencyList), sum(googleFrequencyList), sum(mmFrequencyList), sum(mmFilteredFrequencyList))
##print(sum(w2vFPList), sum(d2vFPList), sum(i2b2FPList), sum(googleFPList), sum(mmFPList), sum(mmFilteredFPList))
RAAr = computeRecall(RAAFrequencyList,denom)
Forumr = computeRecall(ForumFrequencyList,denom)
EMSr = computeRecall(EMSFrequencyList,denom)
I2B2r = computeRecall(I2B2FrequencyList,denom)
MIMIC3r  = computeRecall(MIMIC3FrequencyList,denom)
Medr =  computeRecall(MedFrequencyList,denom)
MedEMSr = computeRecall(MedEMSFrequencyList, denom)

RAAp = computePrecision(RAAFrequencyList, RAAFPList)
Forump = computePrecision(ForumFrequencyList, ForumFPList)
EMSp = computePrecision(EMSFrequencyList, EMSFPList)
I2B2p = computePrecision(I2B2FrequencyList, I2B2FPList)
MIMIC3p = computePrecision(MIMIC3FrequencyList, MIMIC3FPList)
Medp = computePrecision(MedFrequencyList, MedFPList)
MedEMSp = computePrecision(MedEMSFrequencyList, MedEMSFPList)

filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/RecallPrecision/Results_wordEmbedding.txt"
wfile = open (filePath, "w")
##wfile.write(str(sum(w2vFrequencyList))+ "\t" + str(sum(d2vFrequencyList))+ "\t" + str(sum(i2b2FrequencyList))+ "\t" + str(sum(googleFrequencyList))+ "\t" +  str(sum(mmFrequencyList)) +"\n")
##wfile.write(str(sum(w2vFPList))+ "\t" + str(sum(d2vFPList))+ "\t" + str(sum(i2b2FPList))+ "\t" + str(sum(googleFPList))+ "\t" +  str(sum(mmFPList)) +"\n")

##wfile.write("ModelName\tWord2Vec\tDoc2Vec\tI2B2\tGoogle\tMMoutput\tMMoutputFiltST\tMM_ManualList\tMMList\tMMListFiltST\n")
wfile.write("ModelName\tRAA\tForum\tEMS\tI2B2\tMIMIC3\tMed\tMedEMS\n")
wfile.write("Recall\t"+str(RAAr)+"\t"+str(Forumr)+"\t"+str(EMSr)+"\t"+str(I2B2r)+"\t"+str(MIMIC3r)+"\t"+str(Medr)+"\t"+str(MedEMSr)+"\n")
wfile.write("Precision\t"+str(RAAp)+"\t"+str(Forump)+"\t"+str(EMSp)+"\t"+str(I2B2p)+"\t"+str(MIMIC3p)+"\t"+str(Medp)+"\t"+str(MedEMSp)+"\n")

wfile.close()


##wfile.write("Recall\t"+str(round(sum(w2vFrequencyList)/float(denom),4))+ "\t" + str(round(sum(d2vFrequencyList)/float(denom),4))+ "\t" + str(round(sum(i2b2FrequencyList)/float(denom),4))+ "\t" + str(round(sum(googleFrequencyList)/float(denom),4))+"\t"+ str(round(sum(mmFrequencyList)/float(denom),4))+ "\t" + str(round(sum(mm_FiltST_FrequencyList)/float(denom),4))+ "\t" +  str(round(sum(mmManFrequencyList)/float(denom),4)) + "\t" +str(round(sum(mmExtConceptFrequencyList)/float(denom),4))+ "\t" + str(round(sum(mm_FiltST_ExtConceptFrequencyList)/float(denom),4))+"\n")
# wfile.write("Recall\t" + str(round(sum(i2b2FrequencyList)/float(denom),4))+ "\t" + str(round(sum(googleFrequencyList)/float(denom),4))+"\t"+ str(round(sum(mmFrequencyList)/float(denom),4))+ "\t" + str(round(sum(mm_FiltST_FrequencyList)/float(denom),4))+ "\t" +  str(round(sum(mmManFrequencyList)/float(denom),4)) + "\t" +str(round(sum(mmExtConceptFrequencyList)/float(denom),4))+ "\t" + str(round(sum(mm_FiltST_ExtConceptFrequencyList)/float(denom),4))+"\n")


# ##w2vp = computePrecision(w2vFrequencyList, w2vFPList)
# ##d2vp = computePrecision(d2vFrequencyList, d2vFPList)
# i2b2p = computePrecision(i2b2FrequencyList, i2b2FPList)
# gp = computePrecision(googleFrequencyList, googleFPList)
# mmManp = computePrecision(mmManFrequencyList, mmManFPList)
# ##mmFilteredp = computePrecision(mmFilteredFrequencyList, mmFilteredFPList)
# mmp = computePrecision(mmFrequencyList, mmFPList)
# mm_FiltST_p = computePrecision(mm_FiltST_FrequencyList, mm_FiltST_FPList)
# mmListp = computePrecision(mmExtConceptFrequencyList, mmExtConceptFPList)
# mm_FiltST_Listp = computePrecision(mm_FiltST_ExtConceptFrequencyList, mm_FiltST_ExtConceptFPList)

# ##wfile.write("ModelName\tWord2Vec\tDoc2Vec\tI2B2\tGoogle\tMMoutput\tMMoutputFiltST\tMM_ManualList\tMMList\tMMListFiltST\n")
# wfile.write("ModelName\tI2B2\tGoogle\tMMoutput\tMMoutputFiltST\tMM_ManualList\tMMList\tMMListFiltST\n")
# ##wfile.write("Precision\t"+str(w2vp)+ "\t" + str(d2vp)+ "\t" + str(i2b2p)+ "\t" + str(gp)+ "\t" + str(mmp)+ "\t" + str(mm_FiltST_p)+ "\t" + str(mmManp) + "\t"+ str(mmListp)+ "\t" + str(mm_FiltST_Listp)+ "\n")
# wfile.write("Precision\t"+ str(i2b2p)+ "\t" + str(gp)+ "\t" + str(mmp)+ "\t" + str(mm_FiltST_p)+ "\t" + str(mmManp) + "\t"+ str(mmListp)+ "\t" + str(mm_FiltST_Listp)+ "\n")

