# -*- coding: utf-8 -*-
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
#from nltk.tag import pos_tag

def textParsing(fileList1):
    sentsList0=[] #temp
    for files in fileList1:
        with open(files, 'r') as in_file:
            text = in_file.read()
            #wordsList0.append(nltk.word_tokenize(text))
            sentsList0.append(sent_tokenize(text))
        in_file.close()
    #print(sentsList[1])

    sentences0=''
    
    for i in range(len(sentsList0)):
        sentences0=sentences0+str(sentsList0[i])
        
    sentences1=sentences0.replace("[","")
    sentences2=sentences1.replace("]","")
    sentences3=sentences2.replace("'","")
    sentences4=sentences3.replace(".,",".")
    sentences5=sentences4.replace(".",". ")
    
    return sentences5

def listToText(sentenceList):
    sentences0=''
    
    for i in range(len(sentenceList)):
        sentences0=sentences0+str(sentenceList[i])
        
    sentences1=sentences0.replace("[","")
    sentences2=sentences1.replace("]","")
    sentences3=sentences2.replace("'","")
    sentences4=sentences3.replace(".,",".")
    sentences5=sentences4.replace(".",". ")
    
    return sentences5

