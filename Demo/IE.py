from pymetamap import MetaMap
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import time
import re

class PatientStatus(object):
    def __init__(self, name, binary, value = -1, content = ''):
        self.name = name
        self.binary = binary
        self.value = value
        self.content = content

class ConceptExtraction(object):
    def __init__(self, List_route, Status = dict()):
        '''
        list_route: route of the list file
        extend list: a .csv file, two colum: required concepts and its cuis.
        status: a dict, keys are concepts, values are corresponding informations. 
                Indicates the default status of the patient.
        self.CUIs: list of the requied CUIs
        self.CUI2Concept: mapping the CUIs to the concepts
        self.Status: dict to store the information
        self.mm: MetaMap object
        self.R_range: range of value retrival in the text, default: 14
        self.pattern: pattern of the requied value
        '''
        extended_concept_list = pd.read_csv(List_route)
        self.CUIs = [item for item in extended_concept_list['CUI']]
        self.CUI2Concept = {}
        for idx,item in enumerate(extended_concept_list['Required Concept']):
            if item > 0:
                temp = item
                self.CUI2Concept[self.CUIs[idx]] = temp.lower()
            else:
                self.CUI2Concept[self.CUIs[idx]] = temp.lower()
        self.Status = Status
        self.mm = MetaMap.get_instance('/Users/sileshu/Downloads/public_mm/bin/metamap16')
        self.R_range = 14
        self.pattern = "\d+\.?\d*"
        self.ex_extend_cons = []
        self.raw_ex_cons = []
        
    def StatusInit(self):
        '''
        if don't have a defined initial status, this function can generate a default status from the concept list
        all the binary status are defined as False initially
        '''
        for item in list(set(self.CUI2Concept.values())):
            self.Status[item] = PatientStatus(item, False)
    
    def ConceptExtract(self, sent_text):
        '''
        sent_text: a list of string
        '''
        concepts,_ = self.mm.extract_concepts(sent_text,word_sense_disambiguation=True,\
                                     ignore_stop_phrases=True)
        for concept in concepts:
            normalized_trigger_name = concept[6].split('-')[3].strip('"').lower()
            # last part of "trigger" field, 1 means negation is detected
            negation = concept[6].split('-')[-1].rstrip(']') == '0' # negation = False if negation is detected
            CUI = concept[4]
            posi_info = concept[8].replace(';',',').strip('[]').split('],[')
            for i in range(len(posi_info)):
                self.raw_ex_cons.append(concept)
                if CUI in self.CUIs:
                    if ',' in posi_info[i]:
                        position = posi_info[i].split(',')[-1].split('/')
                    else:
                        position = posi_info[i].split('/')
                    beginPt = int(position[0])
                    length = int(position[1])
                    strPiece = sent_text[0][beginPt+length:beginPt+length+self.R_range]
                    mapped_concept = self.CUI2Concept[CUI]
                    self.ex_extend_cons.append(mapped_concept)
                    if mapped_concept in self.Status:
                        self.Status[mapped_concept].binary = negation
                        self.Status[mapped_concept].content = normalized_trigger_name
                        value = re.findall(self.pattern,strPiece)
                        if len(value) > 0:
                            if mapped_concept == 'blood pressure':
                                self.Status[mapped_concept].value = (value[0]+'/'+value[1])
                            else:
                                self.Status[mapped_concept].value = value[0]

if __name__ == "__main__":
	sents = ["this is paramedic Smith we are transporting a 36 year old male complaining of shortness of breath he also complains of non radiating sharp chest pain on expiration symptoms began 1 hour ago after exercising he attempted to use his inhaler without relief he does have a history of asthma this patient was found with obvious respiratory difficulty he is awake and appropriate on exam we find lung sounds with wheezes in Upper lobes bilaterally and quiet lower lobes bilaterally initial Vital Signs BP 162 over 94 pulse 120 for respirations 36 skin is warm and moistpulse oximetry 92% on 15 LPM by non-rebreather ECG sinus tach without a activate we've treated with high-flow oxygen followed by one dose of Proventil and atrovent by small volume nebulizer the patient has improved and now has respirations of 24 flight wheezes in all Fields Pulse oximetry 96% on 6 LPM by nebulizer we plan to begin a second dose of Proventil by nebulizer and transport to your ETA ETA is 7 minutes do you have any questions"]
	t = ConceptExtraction("/Users/sileshu/Desktop/Extended_Concept_List.csv")
	t.StatusInit()
	t.ConceptExtract(sents)
	for item in t.Status:
	    print (t.Status[item].name)
	    print (t.Status[item].binary)
	    print (t.Status[item].value)
	    print (t.Status[item].content)