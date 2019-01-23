import behaviours as be
import py_trees
from pandas import Series, DataFrame
import pandas as pd
import re
from py_trees.blackboard import Blackboard
import ConceptExtract as CE
import time

def post_tick_handler(behaviour_tree):
    blackboard = Blackboard()
    print blackboard.protocol
    if hasattr(blackboard, 'action'):
        print blackboard.action
		
# action leaves
class InformationGathering(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'InformationGathering'):
        super(InformationGathering, self).__init__(name)
        
    def setup(self, unused_timeout = 15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        self.ce1 = CE.ConceptExtractor("Concept_List_1.csv")
        self.ce1.StatusInit()
        self.ce2 = CE.ConceptExtractor("Concept_List_2.csv")
        self.ce2.StatusInit()
        return True
        
    
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

    


class TextCollection(py_trees.behaviour.Behaviour):
    def __init__(self, name = 'TextCollection'):
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
        self.sent_text.append(raw_input("input_text:\n"))
        
        blackboard = Blackboard()
        blackboard.text = self.sent_text
        return py_trees.Status.SUCCESS
		
if __name__ == "__main__":
	root_p = py_trees.composites.Sequence("Root_p")
	selector = py_trees.composites.Selector("ProtocolSelector")
	ChestPain = py_trees.composites.Sequence("ChestPain")
	BleedingControl = py_trees.composites.Sequence("BleedingControl")
	BurnInjury = py_trees.composites.Sequence("BurnInjury")
	BLSCPR = py_trees.composites.Sequence("BLSCPR")
	GTM = py_trees.composites.Sequence("GeneralTraumaManagement")
	RES = py_trees.composites.Sequence("RespiratoryDistress")
	BLSCPR_C = be.BLSCPR_Condition()
	ChestPain_C = be.ChestPain_Condition()
	BC_C = be.Bleeding_Control_Condition()
	BI_C = be.Burn_Injury_Condition()
	GTG_C = be.GeneralTraumaGuideline_Condition()
	RD_C = be.RespiratoryDistress_Condition()
	IG = InformationGathering()
	TC = TextCollection()
	Arb = be.Arbiter()
	GTM.add_children([GTG_C, be.GTM_Action])
	ChestPain.add_children([ChestPain_C, be.ChestPain_Action])
	BleedingControl.add_children([BC_C, be.BC_Action])
	BurnInjury.add_children([BI_C, be.Burn_Action])
	BLSCPR.add_children([BLSCPR_C,be.BLSCPR_Action])
	RES.add_children([RD_C,be.Respiratory_Action])
	root_p.add_children([TC,IG,selector])
	selector.add_children([BLSCPR,BleedingControl,BurnInjury,GTM,RES,ChestPain,Arb])
	behaviour_tree = py_trees.trees.BehaviourTree(root_p)
	behaviour_tree.add_post_tick_handler(post_tick_handler)
	behaviour_tree.setup(15)
	
	behaviour_tree.tick_tock(
	            sleep_ms=50,
	            number_of_iterations=10,
	            pre_tick_handler=None,
	       post_tick_handler=None
	        )


