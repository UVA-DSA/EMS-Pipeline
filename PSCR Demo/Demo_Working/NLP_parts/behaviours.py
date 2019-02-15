import py_trees
from pandas import Series, DataFrame
import pandas as pd
import re
from py_trees.blackboard import Blackboard
import ConceptExtract as CE
import time

# action leaves
class InformationGathering(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'InformationGathering'):
        super(InformationGathering, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        self.ce1 = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/Concept_List_1.csv")
        self.ce1.StatusInit()
        self.ce2 = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/Concept_List_2.csv")
        self.ce2.StatusInit()
        return True
    
    def initialise(self):
        pass
       # self.StartTime = time.time()
        
    
    def update(self):
        blackboard = Blackboard()
        self.ce1.ConceptExtract(blackboard.text)
        blackboard.concepts = self.ce1.concepts
        self.ce1.FirstExtract(blackboard.text)
        blackboard.status1 = self.ce1.Status
        self.ce2.concepts = blackboard.concepts
        self.ce2.SecondExtract(blackboard.text)
        blackboard.status2 = self.ce2.Status
        self.ce1.DisplayStatus()
        self.ce2.DisplayStatus()
        return py_trees.Status.SUCCESS
    
    def terminate(self, new_status):
        pass
    

class TextCollection(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'TextCollection'):
        super(TextCollection, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        #level = raw_input("Please type in your certification(EMT,A,I/P): \n")
        level = 'I/P'
        blackboard = Blackboard()
        blackboard.level = level
        blackboard.tick_num = 0
        blackboard.protocol = "Universal Patient Care"
        self.text = ''
        self.sent_text = []
        return True
    
    def initialise(self):
        self.sent_text = []

    
    def update(self):
        
        if (not q.empty()) and len(self.text) <= 100:
            self.text += q.get()
        if len(self.text) > 100:
            self.sent_text.append(self.text)
            self.text = ''

        blackboard = Blackboard()
        blackboard.text = self.sent_text
        
        #self.sent_text.append(raw_input("input_text:\n"))

        #blackboard = Blackboard()
        #blackboard.text = self.sent_text
        return py_trees.Status.SUCCESS
    
    def terminate(self, new_status):
        pass
		
# protocols' condition
class BLSCPR_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'BLSCPR_Condition'):
        super(BLSCPR_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/BLSCPR.csv")
        self.ce.StatusInit()
        blackboard.BLSCPR_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.BLSCPR_Status = self.ce.Status
        if blackboard.status1['pulse'].binary == False and \
        blackboard.status1['breath'].binary == False:
            blackboard.Protocol = "BLS CPR"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class Bleeding_Control_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Bleeding_Control_Condition'):
        super(Bleeding_Control_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/BleedingControl.csv")
        self.ce.StatusInit()
        blackboard.BC_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.BC_Status = self.ce.Status
        if blackboard.status1['bleed'].binary == True:
            blackboard.Protocol = "Bleeding Control"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class AlteredMentalStatus_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'AlteredMentalStatus_Condition'):
        super(AlteredMentalStatus_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/AMS.csv")
        self.ce.StatusInit()
        blackboard.AMS_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.AMS_Status = self.ce.Status
        if blackboard.status1['breath'].binary == True and blackboard.status1['conscious'].binary == False \
        and blackboard.status1['pulse'].binary == True:
            blackboard.protocol = "Altered Mental Status"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class Burn_Injury_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'Burn_Injury_Condition'):
        super(Burn_Injury_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/Burn.csv")
        self.ce.StatusInit()
        blackboard.Burn_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.Burn_Status = self.ce.Status
        if blackboard.status1['wound'].binary == True and blackboard.status2['burn'].binary == True:
            blackboard.protocol = "Burn Injury"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class GeneralTraumaGuideline_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'GeneralTraumaGuideline_Condition'):
        super(GeneralTraumaGuideline_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/GeneralTrauma.csv")
        self.ce.StatusInit()
        blackboard.GT_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.GT_Status = self.ce.Status
        if blackboard.status1['wound'].binary == True and blackboard.status2['burn'].binary == False:
            blackboard.protocol = "General Trauma Guideline"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
        
class RespiratoryDistress_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'RespiratoryDistress_Condition'):
        super(RespiratoryDistress_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/Respiratory.csv")
        self.ce.StatusInit()
        blackboard.Res_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.Res_Status = self.ce.Status
        if blackboard.status1['breath'].binary == False:
            blackboard.protocol = "Respiratory Distress"
            return py_trees.Status.SUCCESS
        if len(blackboard.status1['breath'].value) > 0:
            if int(blackboard.status1['breath'].value) > 30 or int(blackboard.status1['breath'].value) < 10:
                blackboard.protocol = "Respiratory Distress"
                return py_trees.Status.SUCCESS
        if len(blackboard.status1['spo2'].value) > 0:
            if int(blackboard.status1['spo2'].value.replace('%','')) < 70:
                blackboard.protocol = "Respiratory Distress"
                return py_trees.Status.SUCCESS
        if len(blackboard.status1['etco2'].value) >0:
            if int(blackboard.status1['etco2'].value) > 45 or int(blackboard.status1['etco2'].value) < 35:
                blackboard.protocol = "Respiratory Distress"
                return py_trees.Status.SUCCESS
        return py_trees.Status.FAILURE
        
class ChestPain_Condition(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'ChestPain_Condition'):
        super(ChestPain_Condition, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        blackboard = Blackboard()
        self.ce = CE.ConceptExtractor("/home/hotaki/PyQt/NLP_parts/ChestPain.csv")
        self.ce.StatusInit()
        blackboard.CP_Status = self.ce.Status
        return True
    
    def initialise(self):
        pass
    
    def update(self):
        blackboard = Blackboard()
        self.ce.concepts = blackboard.concepts
        self.ce.SecondExtract(blackboard.text)
        blackboard.CP_Status = self.ce.Status
        if blackboard.status1['pain'].binary == True and \
        ('chest' in blackboard.status1['pain'].content or \
         'chest' in blackboard.status2['pain region'].content or\
         'chest' in blackboard.status2['pain region'].value):
            blackboard.protocol = "Chest Pain"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE
