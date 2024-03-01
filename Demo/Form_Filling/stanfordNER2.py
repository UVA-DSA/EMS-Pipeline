# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.tag import StanfordNERTagger
from collections import Counter

from textParse2 import listToText


#for abs path, add "/home/arif/Desktop/stanfordNER/"

st3=StanfordNERTagger('./stanfordNER/stanford-ner-2018-02-27/classifiers/english.all.3class.distsim.crf.ser.gz',
                    './stanfordNER/stanford-ner-2018-02-27/stanford-ner.jar',
                    encoding='utf-8')
st4=StanfordNERTagger('./stanfordNER/stanford-ner-2018-02-27/classifiers/english.conll.4class.distsim.crf.ser.gz',
                    './stanfordNER/stanford-ner-2018-02-27/stanford-ner.jar',
                    encoding='utf-8')
st7=StanfordNERTagger('./stanfordNER/stanford-ner-2018-02-27/classifiers/english.muc.7class.distsim.crf.ser.gz',
                    './stanfordNER/stanford-ner-2018-02-27/stanford-ner.jar',
                    encoding='utf-8')

def posTagger(sentences):
    #print(sentenceList)
    #text=listToText(sentenceList)
    #print(text)
    tokenizedWordlist=word_tokenize(sentences)
    #print(tokenizedWordlist)   
    wordListPOS=pos_tag(tokenizedWordlist)
    #print(wordListPOS)
    return wordListPOS

def getEntity(sentences,choice):
    #print(sentenceList)
    #text=listToText(sentenceList)
    #print(text)
    tokenizedWordlist=word_tokenize(sentences)
    #print(tokenizedWordlist)
    if (choice==3):
        wordListPOS=st3.tag(tokenizedWordlist)
    if (choice==4):
        wordListPOS=st4.tag(tokenizedWordlist)
    if (choice==7):
        wordListPOS=st7.tag(tokenizedWordlist)
    #print(wordListPOS)
    return wordListPOS  
    
    
#h=getEntity('I met Matt at the airport', 7)
