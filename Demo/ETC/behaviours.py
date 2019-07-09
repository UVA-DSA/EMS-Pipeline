import py_trees
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from scipy import spatial
import re
from py_trees.blackboard import Blackboard
import ConceptExtract as CE
import time
from ranking_func import rank
from collections import defaultdict

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# dummy leaves
class dummy(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(dummy, self).__init__(name)

class PROTOCOLi_Check(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'PROTOCOLi Check'):
        super(PROTOCOLi_Check, self).__init__(name)

class PROTOCOLi_Action(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'PROTOCOLi Action'):
        super(PROTOCOLi_Action, self).__init__(name)
        
# behaviors in framework
class InformationGathering(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Information Extraction',\
    slist = "/Users/sileshu/Desktop/BT/concept_list(s&s)_revised.csv",\
    vlist = "/Users/sileshu/Desktop/BT/Concept_List_1.csv",\
    exlist = "/Users/sileshu/Desktop/BT/CLfromVt.csv",\
    intlist = "/Users/sileshu/Desktop/BT/concept_list(interventions).csv", inC = None):
        super(InformationGathering, self).__init__(name)
        self.slist = slist
        self.vlist = vlist
        self.exlist = exlist
        self.intlist= intlist
        self.inC = inC
        
    def setup(self, unused_timeout = 15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        '''
        self.ce1 = CE.ConceptExtractor("Concept_List_1.csv")
        self.ce1.StatusInit()
        self.ce2 = CE.ConceptExtractor("Concept_List_2.csv")
        self.ce2.StatusInit()
        '''
        vcl = pd.read_csv(self.exlist)
        blackboard = Blackboard()
        self.sce = CE.ConceptExtractor(self.slist)
        self.sce.StatusInit()
        for item in vcl:
            self.sce.SpecificInit(item)
        self.vce = CE.ConceptExtractor(self.vlist)
        #self.vce.StatusInit()
        self.ice = CE.ConceptExtractor(self.intlist)
        self.ice.StatusInit()
        self.vce.StatusInit()
        if self.inC:
            self.vce.ConceptWrapper(self.inC)
        pool = ['Pulse', 'Resp', 'BP', 'GCS', 'Glucose', 'SPO2', 'Pain', 'EKG']
        for item in pool:
            self.vce.SpecificInit(item)
        blackboard.Signs = self.sce.Status
        blackboard.Vitals = self.vce.Status
        blackboard.Inters = self.ice.Status
        blackboard.ConcLog = []
        return True
    
    def Vital2Symptom(self):
        '''
        Pulse-85  tachy > 100 extreme tachy >150 bradicardia <= 50
        Resp-16  8 - 24 fast slow normal
        BP-164/101  hyper age + 100 hypo < 70 normal mbp =  (sbp + 2*dbp)/3 
        GCS-15 decreased mental status <= 14
        Glucose-101  hyper >300 hypo <60
        SPO2-100 94 - 99 
        Pain-9 pain severity
        '''
        # Pulse to bradycardia or tachycardia
        if len(self.vce.vtLog['Pulse']) > 0:
            for v in self.vce.vtLog['Pulse']:
                pr = int(v)
                if pr > 100:
                    self.sce.Status['tachycardia'] = CE.PatientStatus('tachycardia',True,str(pr),'Pulse')
                    self.sce.Status['tachycardia'].score = 1000.
                elif pr <= 50:
                    self.sce.Status['bradycardia'] = CE.PatientStatus('bradycardia',True,str(pr),'Pulse')
                    self.sce.Status['bradycardia'].score = 1000.
        # Resp rate symptoms
        if len(self.vce.vtLog['Resp']) > 0:
            for v in self.vce.vtLog['Resp']:
                rr = int(v)
                if rr > 24:
                    self.sce.Status['tachypnea'] = CE.PatientStatus('tachypnea',True,str(rr),'Resp')
                    self.sce.Status['tachypnea'].score = 1000.
                elif rr < 8:
                    self.sce.Status['bradypnea'] = CE.PatientStatus('bradypnea',True,str(rr),'Resp')
                    self.sce.Status['bradypnea'].score = 1000.
                
        # blood pressure symptoms
        if len(self.vce.vtLog['BP']) > 0:
            for v in self.vce.vtLog['BP']:
                if '/' not in v:
                    continue
                bp = [int(n) for n in v.strip().split('/')]
                mbp = (bp[0] + 2 * bp[1])/3
                if bp[0] > 140 and bp[1] > 90:
                    self.sce.Status['hypertension'] = CE.PatientStatus('hypertension',True,str(bp[0]),'BP')
                    self.sce.Status['hypertension'].score = 1000.
                elif mbp < 70:
                    self.sce.Status['hypotension'] = CE.PatientStatus('hypotension',True,str(mbp),'BP')
                    self.sce.Status['hypotension'].score = 1000.
        
        # GCS symptoms
        if len(self.vce.vtLog['GCS']) > 0:
            for v in self.vce.vtLog['GCS']:
                gcs = int(v)
                if gcs < 15:
                    self.sce.Status['decreased mental status'] = CE.PatientStatus('decreased mental status',True,str(gcs),'GCS')
                    self.sce.Status['decreased mental status'].score = 1000.
                
        # glucose symptoms
        if len(self.vce.vtLog['Glucose']) > 0:
            for v in self.vce.vtLog['Pulse']:
                glu = int(v)
                if glu > 300:
                    self.sce.Status['hyperglycemia'] = CE.PatientStatus('hyperglycemia',True,str(glu),'Glucose')
                    self.sce.Status['hyperglycemia'].score = 1000.
                elif glu < 60:
                    self.sce.Status['hypoglycemia'] = CE.PatientStatus('hypoglycemia',True,str(glu),'Glucose')
                    self.sce.Status['hypoglycemia'].score = 1000.
                
        # spo2 symptoms
        if len(self.vce.vtLog['SPO2']) > 0:
            for v in self.vce.vtLog['SPO2']:
                spo2 = int(v)
                if spo2 < 94:
                    self.sce.Status['hypoxemia'] = CE.PatientStatus('hypoxemia',True,str(spo2),'SPO2')
                    self.sce.Status['hypoxemia'].score = 1000.
                
        # pain
        if len(self.vce.vtLog['Pain']) > 0:
            for v in self.vce.vtLog['Pain']:
                ps = int(v)
                self.sce.Status['pain severity'] = CE.PatientStatus('pain severity',True,str(ps),'Pain')
                self.sce.Status['pain severity'].score = 1000.
        
        # EKG symptoms
        if len(self.vce.vtLog['EKG']) > 0:
            for v in self.vce.vtLog['Pulse']:
                ekg = v
                if "Sinus_Arrhythmia" in ekg:
                    self.sce.Status['dysrhythmia'] = CE.PatientStatus('dysrhythmia',True,ekg,'EKG')
                    self.sce.Status['dysrhythmia'].score = 1000.
                else:
                    self.sce.Status['dysrhythmia'] = CE.PatientStatus('dysrhythmia',False,ekg,'EKG')
                    self.sce.Status['dysrhythmia'].score = 1000.
                if "_Bradycardia" in ekg:
                    self.sce.Status['bradycardia'] = CE.PatientStatus('bradycardia',True,ekg,'EKG')
                    self.sce.Status['bradycardia'].score = 1000.
                if "_Tachycardia" in ekg:
                    self.sce.Status['tachycardia'] = CE.PatientStatus('tachycardia',True,ekg,'EKG')
                    self.sce.Status['tachycardia'].score = 1000.
  
        return True
        
    def update(self):
        blackboard = Blackboard()
        self.sce.ConceptExtract(blackboard.text)
        if blackboard.inC:
            self.vce.ConceptWrapper(blackboard.inC)
        blackboard.concepts = self.sce.concepts
        blackboard.confi = self.sce.scores
        self.sce.FirstExtract(blackboard.text, blackboard.tick_num)
        self.Vital2Symptom()
        blackboard.Signs = self.sce.Status
        
        #self.vce.concepts = blackboard.concepts
        #self.vce.scores = self.sce.scores
        #self.vce.FirstExtract(blackboard.text, blackboard.tick_num)
        blackboard.Vitals = self.vce.Status
        #self.sce.DisplayStatus()
        self.ice.concepts = blackboard.concepts
        self.ice.scores = self.sce.scores
        self.ice.FirstExtract(blackboard.text, blackboard.tick_num)
        blackboard.Inters = self.ice.Status
       # self.ice.DisplayStatus()
    #    blackboard.ConcLog += self.sce.Log + self.vce.Log
        return py_trees.Status.SUCCESS
        
class IG(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'IG',\
    slist = "/Users/sileshu/Desktop/BT/concept_list(s&s)_revised.csv",\
    vlist = "/Users/sileshu/Desktop/BT/Concept_List_1.csv",\
    exlist = "/Users/sileshu/Desktop/BT/CLfromVt.csv",\
    intlist = "/Users/sileshu/Desktop/BT/concept_list(interventions).csv", \
    aio_ss = "/Users/sileshu/Downloads/renewconceptlists/All_In_One_signs&symptoms.xlsx",\
    aio_int = "/Users/sileshu/Downloads/renewconceptlists/All_In_One_interventions.xlsx",\
    inC = None, neg_res = None, WDistance = False, aio_only = False):
        super(IG, self).__init__(name)
        self.slist = slist
        self.vlist = vlist
        self.exlist = exlist
        self.intlist= intlist
        self.aio_ss = aio_ss
        self.aio_int = aio_int
        self.inC = inC
        self.neg_res = neg_res
        self.WDistance = WDistance
        self.aio_only = aio_only
        
    def setup(self, unused_timeout = 15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        '''
        self.ce1 = CE.ConceptExtractor("Concept_List_1.csv")
        self.ce1.StatusInit()
        self.ce2 = CE.ConceptExtractor("Concept_List_2.csv")
        self.ce2.StatusInit()
        '''
        vcl = pd.read_csv(self.exlist)
        blackboard = Blackboard()
        self.sce = CE.CEWithoutMM(self.slist, self.intlist,\
        AllinOne_SS = self.aio_ss, AllinOne_Int = self.aio_int,\
         neg_res = self.neg_res, WDistance = self.WDistance, aio_only = self.aio_only)
        self.sce.StatusInit()
        for item in vcl:
            self.sce.SpecificInit(item)
        self.vce = CE.ConceptExtractor(self.vlist)
        #self.vce.StatusInit()
        #self.ice = CE.ConceptExtractor(self.intlist)
        #self.ice.StatusInit()
        if self.inC:
            self.vce.ConceptWrapper(self.inC)
        blackboard.Signs = self.sce.Status
        blackboard.Vitals = self.vce.Status
        blackboard.Inters = self.sce.Interventions
        blackboard.ConcLog = []
        return True
    
    def Vital2Symptom(self):
        '''
        Pulse-85  tachy > 100 extreme tachy >150 bradicardia <= 50
        Resp-16  8 - 24 fast slow normal
        BP-164/101  hyper age + 100 hypo < 70 normal mbp =  (sbp + 2*dbp)/3 
        GCS-15 decreased mental status <= 14
        Glucose-101  hyper >300 hypo <60
        SPO2-100 94 - 99 
        Pain-9 pain severity
        '''
        # Pulse to bradycardia or tachycardia
        if len(self.vce.vtLog['Pulse']) > 0:
            for v in self.vce.vtLog['Pulse']:
                pr = int(v)
                if pr > 100:
                    self.sce.Status['tachycardia'] = CE.PatientStatus('tachycardia',True,str(pr),'Pulse')
                    self.sce.Status['tachycardia'].score = 1000.
                elif pr <= 50:
                    self.sce.Status['bradycardia'] = CE.PatientStatus('bradycardia',True,str(pr),'Pulse')
                    self.sce.Status['bradycardia'].score = 1000.
        # Resp rate symptoms
        if len(self.vce.vtLog['Resp']) > 0:
            for v in self.vce.vtLog['Resp']:
                rr = int(v)
                if rr > 24:
                    self.sce.Status['tachypnea'] = CE.PatientStatus('tachypnea',True,str(rr),'Resp')
                    self.sce.Status['tachypnea'].score = 1000.
                elif rr < 8:
                    self.sce.Status['bradypnea'] = CE.PatientStatus('bradypnea',True,str(rr),'Resp')
                    self.sce.Status['bradypnea'].score = 1000.
                
        # blood pressure symptoms
        if len(self.vce.vtLog['BP']) > 0:
            for v in self.vce.vtLog['BP']:
                if '/' not in v:
                    continue
                bp = [int(n) for n in v.strip().split('/')]
                mbp = (bp[0] + 2 * bp[1])/3
                if bp[0] > 140 and bp[1] > 90:
                    self.sce.Status['hypertension'] = CE.PatientStatus('hypertension',True,str(bp[0]),'BP')
                    self.sce.Status['hypertension'].score = 1000.
                elif mbp < 70:
                    self.sce.Status['hypotension'] = CE.PatientStatus('hypotension',True,str(mbp),'BP')
                    self.sce.Status['hypotension'].score = 1000.
        
        # GCS symptoms
        if len(self.vce.vtLog['GCS']) > 0:
            for v in self.vce.vtLog['GCS']:
                gcs = int(v)
                if gcs < 15:
                    self.sce.Status['decreased mental status'] = CE.PatientStatus('decreased mental status',True,str(gcs),'GCS')
                    self.sce.Status['decreased mental status'].score = 1000.
                
        # glucose symptoms
        if len(self.vce.vtLog['Glucose']) > 0:
            for v in self.vce.vtLog['Pulse']:
                glu = int(v)
                if glu > 300:
                    self.sce.Status['hyperglycemia'] = CE.PatientStatus('hyperglycemia',True,str(glu),'Glucose')
                    self.sce.Status['hyperglycemia'].score = 1000.
                elif glu < 60:
                    self.sce.Status['hypoglycemia'] = CE.PatientStatus('hypoglycemia',True,str(glu),'Glucose')
                    self.sce.Status['hypoglycemia'].score = 1000.
                
        # spo2 symptoms
        if len(self.vce.vtLog['SPO2']) > 0:
            for v in self.vce.vtLog['SPO2']:
                spo2 = int(v)
                if spo2 < 94:
                    self.sce.Status['hypoxemia'] = CE.PatientStatus('hypoxemia',True,str(spo2),'SPO2')
                    self.sce.Status['hypoxemia'].score = 1000.
                
        # pain
        if len(self.vce.vtLog['Pain']) > 0:
            for v in self.vce.vtLog['Pain']:
                ps = int(v)
                self.sce.Status['pain severity'] = CE.PatientStatus('pain severity',True,str(ps),'Pain')
                self.sce.Status['pain severity'].score = 1000.
        
        # EKG symptoms
        if len(self.vce.vtLog['EKG']) > 0:
            for v in self.vce.vtLog['Pulse']:
                ekg = v
                if "Sinus_Arrhythmia" in ekg:
                    self.sce.Status['dysrhythmia'] = CE.PatientStatus('dysrhythmia',True,ekg,'EKG')
                    self.sce.Status['dysrhythmia'].score = 1000.
                else:
                    self.sce.Status['dysrhythmia'] = CE.PatientStatus('dysrhythmia',False,ekg,'EKG')
                    self.sce.Status['dysrhythmia'].score = 1000.
                if "_Bradycardia" in ekg:
                    self.sce.Status['bradycardia'] = CE.PatientStatus('bradycardia',True,ekg,'EKG')
                    self.sce.Status['bradycardia'].score = 1000.
                if "_Tachycardia" in ekg:
                    self.sce.Status['tachycardia'] = CE.PatientStatus('tachycardia',True,ekg,'EKG')
                    self.sce.Status['tachycardia'].score = 1000.
  
        return True
        
    def update(self):
        blackboard = Blackboard()
        #self.ice.ConceptExtract(blackboard.text)
        #blackboard.concepts = self.ice.concepts
        #blackboard.confi = self.sce.scores
        self.sce.CE(blackboard.text, blackboard.tick_num, blackboard.case_num)
        self.Vital2Symptom()
        blackboard.Signs = self.sce.Status
        #self.vce.concepts = blackboard.concepts
        #self.vce.scores = self.sce.scores
        #self.vce.FirstExtract(blackboard.text, blackboard.tick_num)
        blackboard.Vitals = self.vce.Status
        #self.sce.DisplayStatus()
        #self.ice.concepts = blackboard.concepts
        #self.ice.scores = self.sce.scores
        #self.ice.FirstExtract(blackboard.text, blackboard.tick_num)
        blackboard.Inters = self.sce.Interventions
        #self.ice.DisplayStatus()
        #blackboard.ConcLog += self.sce.Log + self.vce.Log
        self.sce.DisplayStatus()
        return py_trees.Status.SUCCESS
        
        
               
class Vectorize(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Vectorize', protocols = '/Users/sileshu/Desktop/BT/ODEMSA_Protocols_weighted.xlsx'):
        super(Vectorize, self).__init__(name)
        self.protocols = protocols
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        PC = dict()
        pro_df = pd.read_excel(self.protocols)
        for line in pro_df.iterrows():
            line_ss = [(i.strip().lower()[:-1],i.strip().lower()[-1]) for i in line[1]['Signs&Symptoms'].split(';')]
            if not pd.isnull(line[1]['Possible signs&symptoms additions']):
                line_ssr = [(i.strip().lower()[:-1],i.strip().lower()[-1]) for i in line[1]['Possible signs&symptoms additions'].split(';')]
            name = line[1]['Protocol']
            PC[name] = line_ss + line_ssr
        self.PV = dict()
        for item in PC:
            vec = list()
            su = 0.
            for i in blackboard.Signs:
                res = 0.
                for j in PC[item]:
                    if i == j[0]:
                        res = 8.**int(j[1])
                        break;
                su += res
                vec.append(res)
            for i in xrange(len(vec)):
                vec[i] = vec[i] / su
            self.PV[item] = vec
        blackboard.PV = self.PV
        return True
        
    
    def update(self):
        blackboard = Blackboard()  
        # mm confidence encoding
        TV = []
        for item in blackboard.Signs:
            if blackboard.Signs[item].binary:
                TV.append(blackboard.Signs[item].score/1000.)
            else:
                TV.append(0.)
        #TV = [blackboard.Status[item].score/1000. for item in blackboard.Status]
        
        # one-hot encoding
        #TV = [float(blackboard.Signs[item].binary) for item in blackboard.Signs]
        blackboard.TV = TV
        maxsim = 0
        result = ''
        blackboard.ranking = []
        for key in self.PV:
            sim = 1 - spatial.distance.cosine(TV,self.PV[key])
            blackboard.ranking.append((key,sim))
            if sim > maxsim:
                maxsim = sim
                result = key
        blackboard.protocol = result
        blackboard.candi,blackboard.pos = rank(blackboard.ranking)[0],rank(blackboard.ranking)[1]
        return py_trees.Status.SUCCESS

class ProtocolSelector(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Protocol Selector'):
        super(ProtocolSelector, self).__init__(name)
    
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        blackboard.protocol_flag = dict()
        blackboard.feedback = dict()
        for i in blackboard.PV:
            blackboard.protocol_flag[i] = (False,0.)
        for i in blackboard.Inters:
            blackboard.feedback[i] = 0.
        return True
    
    def update(self):
        blackboard = Blackboard()
        blackboard.protocol_flag = dict()
        blackboard.feedback = dict()
        for i in blackboard.PV:
            blackboard.protocol_flag[i] = (False,0.)
        for i in blackboard.Inters:
            blackboard.feedback[i] = 0.
        num = sum(blackboard.pos[:3])
        for idx,item in enumerate(blackboard.candi):
            if idx < 3:
                blackboard.protocol_flag[item] = (True,blackboard.pos[idx]/num)
        return py_trees.Status.SUCCESS

class TextCollection(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Speech To Text Conversion'):
        super(TextCollection, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        #level = raw_input("Please type in your certification(EMT,A,I/P): \n")
        level = 'I/P'
        blackboard = Blackboard()
        #blackboard.action = []
        blackboard.level = level
        blackboard.tick_num = 0
        blackboard.protocol = "Universal Patient Care"
        self.text = ''
        self.sent_text = []
        return True
    
    def initialise(self):
        self.sent_text = []

    
    def update(self):
        '''
        if (not q.empty()) and len(self.text) <= 100:
            self.text += q.get()
        if len(self.text) > 100:
            self.sent_text.append(self.text)
            self.text = ''
        blackboard = Blackboard()
        blackboard.text = self.sent_text
        '''
        '''
        self.sent_text.append(raw_input("input_text:\n"))
        
        blackboard = Blackboard()
        blackboard.text = self.sent_text
        '''
        return py_trees.Status.SUCCESS
 
# protocols' checker
class ChestPain_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'ChestPain_Checker'):
        super(ChestPain_Checker, self).__init__(name)
        self.key = 'Medical - Chest Pain - Cardiac Suspected'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class AbdoPain_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AbdoPain_Checker'):
        super(AbdoPain_Checker, self).__init__(name)
        self.key = 'Medical - Abdominal Pain'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
            
class Behavioral_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Behavioral_Checker'):
        super(Behavioral_Checker, self).__init__(name)
        self.key = 'General - Behavioral/Patient Restraint'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
            
class PainCtrl_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'PainCtrl_Checker'):
        super(PainCtrl_Checker, self).__init__(name)
        self.key = 'General - Pain Control'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE 
            
class Seizure_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Seizure_Checker'):
        super(Seizure_Checker, self).__init__(name)
        self.key = 'Medical - Seizure'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class Resp_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Resp_Checker'):
        super(Resp_Checker, self).__init__(name)
        self.key = 'Medical - Respiratory Distress/Asthma/COPD/Croup/Reactive Airway'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
            
class AMS_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AMS_Checker'):
        super(AMS_Checker, self).__init__(name)
        self.key = 'Medical - Altered Mental Status'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
            
class Diab_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Diab_Checker'):
        super(Diab_Checker, self).__init__(name)
        self.key = 'Medical - Diabetic - Hypoglycemia'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE       
            
class Overdose_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Overdose_Checker'):
        super(Overdose_Checker, self).__init__(name)
        self.key = 'Medical - Overdose/Poisoning - Opioid'
    
    def update(self):
        blackboard = Blackboard()
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE    
        
# protocols
class ChestPain(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'ChestPain'):
        super(ChestPain, self).__init__(name)
        self.key = 'Medical - Chest Pain - Cardiac Suspected'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        if blackboard.candi[0] == self.key and blackboard.Vitals['GCS'].value == '15':
            blackboard.feedback['aspirin'] += self.posi
        # consider add dependency on iv
        if blackboard.Signs['substance abuse history'].binary:
            blackboard.feedback['midazolam'] += self.posi * blackboard.Signs['substance abuse history'].score / 1000.
           # blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['substance abuse history'].score / 1000.
        elif blackboard.Signs['abuse of substance'].binary:
            blackboard.feedback['midazolam'] += self.posi * blackboard.Signs['abuse of substance'].score / 1000.
          #  blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['abuse of substance'].score / 1000.
        if 'STEMI' in blackboard.Vitals['EKG'].value and blackboard.Vitals['Pain'].binary \
        and blackboard.Vitals['BP'].binary and int(blackboard.Vitals['BP'].value.strip().split('/')[0]) > 100:
            blackboard.feedback['nitroglycerin'] += self.posi * blackboard.Vitals['Pain'].score / 1000. * blackboard.Vitals['BP'].score / 1000.
        if  blackboard.Vitals['Pain'].binary and blackboard.Inters['nitroglycerin'].binary:
          #  blackboard.feedback['morphine'] += self.posi * blackboard.Vitals['Pain'].score / 1000. * blackboard.Inters['nitroglycerin'].score / 1000.
            blackboard.feedback['fentanyl'] += self.posi * blackboard.Vitals['Pain'].score / 1000. * blackboard.Inters['nitroglycerin'].score / 1000.
        return py_trees.Status.SUCCESS

CP_C = ChestPain_Checker()
CP_A = ChestPain()
CP = py_trees.composites.Sequence("ChestPain",children = [CP_C,CP_A])
            
class AbdoPain(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AbdoPain'):
        super(AbdoPain, self).__init__(name)
        self.key = 'Medical - Abdominal Pain'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        blackboard.feedback['normal saline'] += self.posi
        if blackboard.Signs['nausea'].binary:
            blackboard.feedback['ondansetron'] += self.posi * blackboard.Signs['nausea'].score / 1000.
        elif blackboard.Signs['vomiting'].binary:
            blackboard.feedback['ondansetron'] += self.posi * blackboard.Signs['vomiting'].score / 1000.
        return py_trees.Status.SUCCESS

AP_C = AbdoPain_Checker()
AP_A = AbdoPain()
AP = py_trees.composites.Sequence("AbdoPain",children = [AP_C,AP_A])      
            
class Behavioral(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Behavioral_Checker'):
        super(Behavioral, self).__init__(name)
        self.key = 'General - Behavioral/Patient Restraint'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['combative'].binary:
            s = blackboard.Signs['combative'].score / 1000.
            if blackboard.Signs['hypoxemia'].binary:
                blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000. * s
            blackboard.feedback['physical restraint'] += self.posi * s
            if blackboard.Signs['agitation'].binary:
                blackboard.feedback['midazolam'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
             #   blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
              #  blackboard.feedback['geodon'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
            blackboard.feedback['transport'] += self.posi * s  
        elif blackboard.Signs['violent'].binary:
            s = blackboard.Signs['violent'].score / 1000.
            if blackboard.Signs['hypoxemia'].binary:
                blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000. * s
            blackboard.feedback['physical restraint'] += self.posi * s
            if not blackboard.Signs['hypotension'].binary and blackboard.Signs['agitation'].binary:
                blackboard.feedback['midazolam'] += self.posi * blackboard.Signs['agitation'].score / 1000. *\
                 blackboard.Signs['hypotension'].score / 1000. * s
              #  blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
            #    blackboard.feedback['geodon'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
            blackboard.feedback['transport'] += self.posi * s
        else:
        #    blackboard.feedback['encourage patient to relax'] += self.posi
            blackboard.feedback['transport'] += self.posi
        return py_trees.Status.SUCCESS

BE_C = Behavioral_Checker()
BE_A = Behavioral()
BE = py_trees.composites.Sequence("Behavioral",children = [BE_C,BE_A])         

            
class PainCtrl(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'PainCtrl'):
        super(PainCtrl, self).__init__(name)
        self.key = 'General - Pain Control'
    
    def update(self):
        '''
        mild pain: ps <= 5
        moderate: ps 6,7
        severe: ps >= 8
        '''
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        if blackboard.Signs['pain severity'].binary:
            s = blackboard.Signs['pain severity'].score / 1000.
            ps = int(blackboard.Signs['pain severity'].value)
            if ps <= 5:
                blackboard.feedback['nitronox'] += self.posi * s
                blackboard.feedback['acetaminophen'] += self.posi * s
                blackboard.feedback['ibuprofen'] += self.posi * s
            elif ps == 6 or ps == 7:
                blackboard.feedback['toradol'] += self.posi * s
            else:
                blackboard.feedback['normal saline'] += self.posi * s
                blackboard.feedback['fentanyl'] += self.posi * s
              #  blackboard.feedback['morphine sulfate'] += self.posi * s
        if blackboard.Signs['nausea'].binary:
            blackboard.feedback['ondansetron'] += self.posi * blackboard.Signs['nausea'].score / 1000.
        elif blackboard.Signs['vomiting'].binary:
            blackboard.feedback['ondansetron'] += self.posi * blackboard.Signs['vomiting'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        return py_trees.Status.SUCCESS

PC_C = PainCtrl_Checker()
PC_A = PainCtrl()
PC = py_trees.composites.Sequence("PainCtrl",children = [PC_C,PC_A]) 
            
class Seizure(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Seizure'):
        super(Seizure, self).__init__(name)
        self.key = 'Medical - Seizure'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['normal saline'] += self.posi
        if not blackboard.Signs['hypotension'].binary and blackboard.Signs['seizure'].binary and blackboard.Vitals['Glucose'].value > 60:
            s = blackboard.Signs['seizure'].score / 1000. * blackboard.Vitals['Glucose'].score / 1000. * blackboard.Signs['hypotension'].score / 1000.
            blackboard.feedback['midazolam'] += self.posi * s
          #  blackboard.feedback['diazepam'] += self.posi * s
        blackboard.feedback['cardiac monitor'] += self.posi
        # recovery position?
        blackboard.feedback['transport'] += self.posi
        return py_trees.Status.SUCCESS

SZ_C = Seizure_Checker()
SZ_A = Seizure()
SZ = py_trees.composites.Sequence("Seizure",children = [SZ_C,SZ_A]) 

class Resp(py_trees.behaviour.Behaviour):
    '''
    Acute respiratory distress syndrome:
    sob: shortness of breath
    rapid breathing: tachypnea
    low bp: hypotension
    confusion: confusion
    '''
    def __init__(self, name = 'Resp'):
        super(Resp, self).__init__(name)
        self.key = 'Medical - Respiratory Distress/Asthma/COPD/Croup/Reactive Airway'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        # have to do?
       # blackboard.feedback['capnography'] += self.posi
       # blackboard.feedback['bronchodilator'] += self.posi 
    #    blackboard.feedback['metered dose inhaler'] += self.posi
        # if respiratory distress
        if blackboard.Signs['shortness of breath'].binary and (blackboard.Signs['hypoxemia'].binary or blackboard.Signs['rhonchi'].binary):
            s = blackboard.Signs['shortness of breath'].score / 1000. *\
             ((blackboard.Signs['hypoxemia'].binary * blackboard.Signs['hypoxemia'].score +\
             blackboard.Signs['rhonchi'].binary * blackboard.Signs['rhonchi'].score) / \
             (blackboard.Signs['hypoxemia'].binary +blackboard.Signs['rhonchi'].binary) / 1000.)
            blackboard.feedback['bag valve mask ventilation'] += self.posi * s
            # bvm ventilation have been done
            if blackboard.Inters['bag valve mask ventilation'].binary:
                blackboard.feedback['endotracheal tube'] += self.posi * s * blackboard.Inters['bag valve mask ventilation'].score / 1000.
            if not blackboard.Signs['hypertension'].binary:
                blackboard.feedback['albuterol'] += self.posi * s * blackboard.Signs['hypertension'].score / 1000.
                blackboard.feedback['ipratropium'] += self.posi * s * blackboard.Signs['hypertension'].score / 1000.
            if blackboard.Signs['wheezing'].binary:
                blackboard.feedback['dexamethasone'] += self.posi * s * blackboard.Signs['wheezing'].score / 1000.
            if blackboard.Signs['hypoxemia'].binary and blackboard.Signs['tachypnea'].binary \
            and blackboard.Vitals['BP'].binary and int(blackboard.Vitals['BP'].value.strip().split('/')[0]) > 90:
                blackboard.feedback['cpap'] += self.posi * s * blackboard.Signs['tachypnea'].score / 1000.
            blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
    
        return py_trees.Status.SUCCESS
        
RE_C = Resp_Checker()
RE_A = Resp()
RE = py_trees.composites.Sequence("Resp",children = [RE_C,RE_A]) 
            
class AMS(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AMS'):
        super(AMS, self).__init__(name)
        self.key = 'Medical - Altered Mental Status'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['normal saline'] += self.posi
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        return py_trees.Status.SUCCESS

AMS_C = AMS_Checker()
AMS_A = AMS()
AMS = py_trees.composites.Sequence("Resp",children = [AMS_C,AMS_A]) 
            
class Diab(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Diab'):
        super(Diab, self).__init__(name)
        self.key = 'Medical - Diabetic - Hypoglycemia'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        if blackboard.Signs['hypoglycemia'].binary:
            s = blackboard.Signs['hypoglycemia'].score / 1000.
            if blackboard.Signs['loss of consciousness'].binary:
                blackboard.feedback['normal saline'] += self.posi * blackboard.Signs['loss of consciousness'].score / 1000. * s
                blackboard.feedback['dextrose'] += self.posi * blackboard.Signs['loss of consciousness'].score / 1000. * s
            else:
                blackboard.feedback['oral glucose'] += self.posi * s
        return py_trees.Status.SUCCESS

DI_C = Diab_Checker()
DI_A = Diab()
DI = py_trees.composites.Sequence("Diab",children = [DI_C,DI_A])    
            
class Overdose(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Overdose'):
        super(Overdose, self).__init__(name)
        self.key = 'Medical - Overdose/Poisoning - Opioid'
    
    def update(self):
        blackboard = Blackboard()
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        if blackboard.Vitals['BP'].binary and int(blackboard.Vitals['BP'].value.strip().split('/')[0]) <= 90:
            blackboard.feedback['normal saline'] += self.posi * blackboard.Vitals['BP'].score / 1000.
        if blackboard.Signs['abuse of substance'].binary:
            blackboard.feedback['narcan'] += self.posi * blackboard.Signs['abuse of substance'].score / 1000.
        elif blackboard.Signs['pin point pupils'].binary:
            blackboard.feedback['narcan'] += self.posi * blackboard.Signs['pin point pupils'].score / 1000.
        elif blackboard.Signs['decreased mental status'].binary:
            blackboard.feedback['narcan'] += self.posi * blackboard.Signs['decreased mental status'].score / 1000.
        elif blackboard.Signs['hypotension'].binary:
            blackboard.feedback['narcan'] += self.posi * blackboard.Signs['hypotension'].score / 1000.
        elif blackboard.Signs['bradypnea'].binary:
            blackboard.feedback['narcan'] += self.posi * blackboard.Signs['bradypnea'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        return py_trees.Status.SUCCESS 
			
OV_C = Overdose_Checker()
OV_A = Overdose()
OV = py_trees.composites.Sequence("Diab",children = [OV_C,OV_A]) 

protocols = py_trees.composites.Parallel('protocols',policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE,\
                                   children = [CP,OV,DI,AMS,RE,SZ,BE,AP])

#py_trees.display.render_dot_tree(OV)

