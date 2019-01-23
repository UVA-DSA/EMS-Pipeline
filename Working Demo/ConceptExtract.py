from pymetamap import MetaMap
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import re


class PatientStatus(object):
    def __init__(self, name, binary, value = '', content = ''):
        self.name = name
        self.binary = binary
        self.value = value
        self.content = content

class ConceptExtractor(object):
    def __init__(self, List_route):
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
        self.R_range = 14
        self.pattern = "\d+\.?\d*"
        #self.ex_extend_cons = []
        #self.raw_ex_cons = []
        
    def StatusInit(self):
        '''
        if don't have a defined initial status, this function can generate a default status from the concept list
        all the binary status are defined as False initially
        '''
        self.Status = dict()
        for item in list(set(self.CUI2Concept.values())):
            if item == 'breath' or item == 'pulse' or item == 'conscious':
                self.Status[item] = PatientStatus(item, True)
            else:
                self.Status[item] = PatientStatus(item, False)
                
            
    
    def ConceptExtract(self, sent_text):
        '''
        sent_text: a list of sent text
        '''
        mm = MetaMap.get_instance('/home/hotaki/Downloads/public_mm/bin/metamap16')
        self.concepts,_ = mm.extract_concepts(sent_text,word_sense_disambiguation=True,\
                                     ignore_stop_phrases=True)
        
    def FirstExtract(self, sent_text):
        for concept in self.concepts:
            normalized_trigger_name = concept[6].split('-')[3].strip('"').lower()
            # last part of "trigger" field, 1 means negation is detected
            negation = concept[6].split('-')[-1].rstrip(']') == '0' # negation = False if negation is detected
            CUI = concept[4]
            posi_info = concept[8].replace(';',',').strip('[]').split('],[')
            preferred_name = concept[3].lower()
            for i in range(len(posi_info)):
                if CUI in self.CUIs:
                    if ',' in posi_info[i]:
                        position = posi_info[i].split(',')[-1].split('/')
                    else:
                        position = posi_info[i].split('/')
                    beginPt = int(position[0])
                    length = int(position[1])
                    if beginPt+length+self.R_range > len(sent_text[0]):
                        latter_strPiece = sent_text[0][beginPt+length-1:len(sent_text[0])]
                    else:
                        latter_strPiece = sent_text[0][beginPt+length-1:beginPt+length+self.R_range]
                    if beginPt-self.R_range < 0:
                        former_strPiece = sent_text[0][0:beginPt]
                    else:
                        former_strPiece = sent_text[0][beginPt-self.R_range:beginPt]
                    mapped_concept = self.CUI2Concept[CUI]
                    if mapped_concept in self.Status:
                        self.Status[mapped_concept].binary = negation
                        self.Status[mapped_concept].content = normalized_trigger_name
                        if mapped_concept == 'age':
                            value = re.findall(self.pattern,former_strPiece)
                        else:
                            value = re.findall(self.pattern,latter_strPiece)
                        if mapped_concept == 'gender' or mapped_concept == 'pain' or mapped_concept == 'trauma':
                            value = normalized_trigger_name
                        if len(value) > 0:
                            if mapped_concept == 'gender' or mapped_concept == 'pain' or mapped_concept == 'trauma':
                                self.Status[mapped_concept].value = normalized_trigger_name
                            elif mapped_concept == 'blood pressure':
                                self.Status[mapped_concept].value = (value[0]+'/'+value[1])
                            elif mapped_concept == 'spo2':
                                self.Status[mapped_concept].value = (value[0]+'%')
                            else:
                                self.Status[mapped_concept].value = value[0]
                                
    def SecondExtract(self, sent_text):
        for concept in self.concepts:
            normalized_trigger_name = concept[6].split('-')[3].strip('"').lower()
            # last part of "trigger" field, 1 means negation is detected
            negation = concept[6].split('-')[-1].rstrip(']') == '0' # negation = False if negation is detected
            CUI = concept[4]
            posi_info = concept[8].replace(';',',').strip('[]').split('],[')
            preferred_name = concept[3].lower()
            for i in range(len(posi_info)):
                if CUI in self.CUIs:
                    if ',' in posi_info[i]:
                        position = posi_info[i].split(',')[-1].split('/')
                    else:
                        position = posi_info[i].split('/')
                    beginPt = int(position[0])
                    length = int(position[1])
                    if beginPt+length+self.R_range > len(sent_text[0]):
                        latter_strPiece = sent_text[0][beginPt+length:len(sent_text[0])]
                    else:
                        latter_strPiece = sent_text[0][beginPt+length:beginPt+length+self.R_range]
                    if beginPt-self.R_range < 0:
                        former_strPiece = sent_text[0][0:beginPt]
                    else:
                        former_strPiece = sent_text[0][beginPt-self.R_range:beginPt]
                    mapped_concept = self.CUI2Concept[CUI]
                    if mapped_concept in self.Status:
                        self.Status[mapped_concept].binary = negation
                        self.Status[mapped_concept].content = normalized_trigger_name
                        if mapped_concept == 'pain severity':
                            value = re.findall(self.pattern,latter_strPiece)
                        else:
                            value = preferred_name
                        if len(value) > 0:
                            if mapped_concept == 'pain severity':
                                self.Status[mapped_concept].value = value[0]
                            else:
                                self.Status[mapped_concept].value = value

    def DisplayStatus(self):
            for item in self.Status:
                if len(self.Status[item].content) > 0:
                    print '('+self.Status[item].name+','+str(self.Status[item].binary)+','+\
                    str(self.Status[item].value)+','+self.Status[item].content+')'
                    '''
                    print 'Concept: ' + self.Status[item].name
                    print 'Presence: ' + str(self.Status[item].binary)
                    print 'value: ' + str(self.Status[item].value)
                    print 'Trigger Text: ' + self.Status[item].content
                    '''

    def ReturnStatus(self):
        return self.Status

    def ReturnStatusList(self):
        StatusList = []
        for item in self.Status:
            if len(self.Status[item].content) > 0:
                StatusList.append('(' + self.Status[item].name + ',' + str(self.Status[item].binary) + ',' + str(self.Status[item].value) + ',' + self.Status[item].content + ')')
        return StatusList


    def ReturnStatusString(self):
        StatusString = ''
        for item in self.Status:
            if len(self.Status[item].content) > 0:
                #print '('+self.Status[item].name+','+str(self.Status[item].binary)+','+\
                #str(self.Status[item].value)+','+self.Status[item].content+')'
                '''
                print 'Concept: ' + self.Status[item].name
                print 'Presence: ' + str(self.Status[item].binary)
                print 'value: ' + str(self.Status[item].value)
                print 'Trigger Text: ' + self.Status[item].content
                '''
                StatusString += '(' + self.Status[item].name + ',' + str(self.Status[item].binary) + ',' + str(self.Status[item].value) + ',' + self.Status[item].content + ')' + '\n'
        return StatusString

    def SpecificInit(self, item):
        '''
        init a specific item in the dictionary
        '''
        if item == 'breath' or item == 'pulse' or item == 'conscious':
            self.Status[item] = PatientStatus(item, True)
        else:
            self.Status[item] = PatientStatus(item, False)

    
