import gensim
import logging
from gensim.models.doc2vec import Doc2Vec

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def isNumber(inputString):
    try:
        float(inputString)
        return True
    except ValueError:
        return False

def loadOriginalConceptList(filepath):
    conceptList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        item = line.split("\t")[0].lower().strip()
        if item not in conceptList:
            conceptList.append(item)
    rfile.close()
    return conceptList

def loadConceptList(filepath):
    conceptList = list()
    rfile = open(filepath, "r")
    for line in rfile:
        items = line.lower().strip().split(" ")
        for item in items:
            if item not in conceptList:
                conceptList.append(item)
    
    rfile.close()
    return conceptList

def createFileName(modelName, n=25):
    filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/Oct2018/PhraseResults/"
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
filePath = "/Users/sarahmasudpreum/Dropbox/NIST_Project/Outputs/newConceptList_Sept2018.txt" # newConceptList_Sept2018 conceptList
# conceptList = loadOriginalConceptList(filePath)
conceptList = loadConceptList(filePath)

print(len(conceptList))
    # Load pre-trained Word2Vec model.
##model = gensim.models.Word2Vec.load("/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/w2v_280_5.model")
##model= Doc2Vec.load("/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/d2v.model") ##d2v_250_10
    
##model= Doc2Vec.load("/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/d2v_250_10.model")
##vl = model.wv.vocab
##model = gensim.models.Word2Vec.load("/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/w2v_250_10.model")
##vl = model.wv.vocab
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sarahmasudpreum/src/word2vecModels/GoogleNews-vectors-negative300.bin', binary=True)  

# word2vec_RAA_Data_150_10 word2vec_Forum_Data_150_10 word2vec_MIMIC3_Data_150_10 word2vec_I2B2_Data_150_10
# word2vec_Med_Data_150_10 word2vec_EMS_Data_150_10 word2vec_MedEMS_Data_150_10
# model = gensim.models.Word2Vec.load('/Users/sarahmasudpreum/Dropbox/NIST_Project/Data/wordEmbeddingModel/Model/word2vec_MedEMS_Data_150_10.model')  

vl = model.vocab ##model.vocab (for google) model.wv.vocab
f1, f2 = createFileName("Google") 
## RAA Forum MIMIC3 I2B2 Med EMS all Google Glove
wfile = open(f1, "w")
wfile2 = open(f2, "w")
candidateList = list()
TOPN = 20

for concept in conceptList:
    if concept in vl:        
        candidateList = model.most_similar(concept, topn = TOPN)
        if len(candidateList) > 0:
            wfile.write(concept+"\t")
            wfile2.write(concept+"\t")
        for item in candidateList:
            if not isNumber(item[0]) and item[0].find("numberToken") is -1: 
                wfile.write(str(item[0])+"\t")
                wfile2.write(str(item[0])+","+str(item[1])+"\t")
        wfile.write("\n")
        wfile2.write("\n")
wfile.close()
wfile2.close()


# Load Google's pre-trained i2b2 model.
##model = gensim.models.KeyedVectors.load_word2vec_format('/Users/sarahmasudpreum/src/word2vecModels/vec_alli2b2Data.bin', binary=True)  

##print("Does it include the stop words like \'a\', \'and\', \'the\'? %d %d %d" % ('a' in model.vocab, 'and' in model.vocab, 'the' in model.vocab))

##model.wv.most_similar(positive=['woman', 'king'], negative=['man'])


