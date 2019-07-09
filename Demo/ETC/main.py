import py_trees
import behaviours_m as be
from py_trees.blackboard import Blackboard
import pandas as pd
from scipy import spatial
import numpy as np
from pandas import DataFrame
import re
import matplotlib.pyplot as plt
import text_clf_utils as utils
import pickle
import textwrap
from ranking_func import rank
from tqdm import tqdm as tqdm
from operator import itemgetter

# function to return Results
def displayRes():
    b = Blackboard()
    protocol_candidates = []
    signs_and_vitals = []
    suggestions = []

    print("===============================================================")

    # Top 3 candidates
    print("\nTop 3 candidates:")
    for c in b.candi[:3]:
        protocol_candidates.append(c)
        print(c)

    # Signs, symptoms and vital
    print("\nSigns, symptoms and vital:")
    for item in b.Vitals:
        if len(b.Vitals[item].content) > 0:
            content = (b.Vitals[item].name, str(b.Vitals[item].binary),
            str(b.Vitals[item].value), b.Vitals[item].content, 
            b.Vitals[item].score, b.Vitals[item].tick)
            signs_and_vitals.append(content)
            print(content)

    for item in b.Signs:
        if len(b.Signs[item].content) > 0:
            content = (b.Signs[item].name, str(b.Signs[item].binary),
            str(b.Signs[item].value), b.Signs[item].content,
            b.Signs[item].score, b.Signs[item].tick)
            signs_and_vitals.append(content)
            print(content)

    # Sort by Tick Number
    signs_and_vitals = sorted(signs_and_vitals, key = itemgetter(5))

    # Suggestions
    print("\nSuggestions:")
    for key in b.feedback:
        if b.feedback[key] > 0.1:
            content = (key, b.feedback[key])
            suggestions.append(content)
            print(content)

    # Create output strings formatted for readibility
    protocol_candidates_str, signs_and_vitals_str, suggestions_str = "", "", ""

    for p in protocol_candidates:
        protocol_candidates_str += str(p) + "\n"

    for sv in signs_and_vitals:
        signs_and_vitals_str += str(sv) + "\n"

    for s in suggestions:
        suggestions_str += str(s) + "\n"


    print("===============================================================")
    return protocol_candidates_str, signs_and_vitals_str, suggestions_str
# param
intermapping= {
    '12-lead ecg':['cardiac monitor'],
    'albuterol':['albuterol'],
    'aspirin':['aspirin'],
    'assist ventilation (bvm)':['bag valve mask ventilation'],
    'capnography (1) first reading':['capnography'],
    'capnography (2) seco':['capnography'],
    'capnography (2) second reading':['capnography'],
    'capnography (3) final reading':['capnography'],
    'cardiac monitor':['cardiac monitor'],
    'nasopharyngeal airway insertio':['nasopharyngeal airway'],
    'cpap':['cpap'],
    'dexamethasone (decadron)':['dexamethasone'],
    'dextrose 10%':['dextrose'],
    'dextrose 25%':['dextrose'],
    'dextrose 50%':['dextrose'],
    'dextrose 5% in 0.45% ns':['dextrose'],
    'duoneb':['albuterol','ipratropium'],
    'fentanyl':['fentanyl'],
    'glucagon':['glucagon'],
    'glutose':['oral glucose'],
    'hospital':['transport'],
    'hospital contact':['transport'],
    'intubation':['endotracheal tube'],
    'ipratropium (atrovent)':['ipratropium'],
    'iv':['normal saline'],
    'magnesium sulfate':['morphine sulfate'],
    'midazolam (versed)':['midazolam'],
    'naloxone (narcan)':['narcan'],
    'nitroglycerine':['nitroglycerine'],
    'normal saline':['normal saline'],
    'ondansetron (zofran)':['ondansetron'],
    'oral glucose':['oral glucose'],
    'oxygen':['oxygen'],
    'restraints':['physical restraint'],
    'suction':['suction the oropharynx','suction the nasopharynx'],
}

EKGdic = {
     '':'',
     'AV_Block_1st_Deg':'AV_Block-1st_Degree',
     'AV_Block_1st_Degree':'AV_Block-1st_Degree',
     'AV_Block_2nd_Degree_Type_1':'AV_Block_2nd_Degree_Type_1',
     'AV_Block_2nd_Degree_Type_2':'AV_Block_2nd_Degree_Type_2',
     'AV_Block_3rd_Degree':'AV_Block_3rd_Degree',
     'Asystole':'Asystole',
     'Artifact':'Artifact',
     'Atrial_Fibrill':'Atrial_Fibrillation',
     'Atrial_Fibrillation':'Atrial_Fibrillation',
     'Atrial_Flutter':'Atrial_Flutter',
     'Juncti':'Junctional_Rhythm',
     'Junctiona':'Junctional_Rhythm',
     'Junctional':'Junctional_Rhythm',
     'Other_(Not_Listed)':'Other_(Not_Listed)',
     'P': 'Paced_Rhythm',
     'PEA':'Pulseless_Electrical_Activity',
     'Pac':'Paced_Rhythm',
     'Paced_Rhythm':'Paced_Rhythm',
     'Premature_Ventricular_Contractions':'Premature_Ventricular_Contractions',
     'Right_Bundle_Branch_Block':'Right_Bundle_Branch_Block',
     'Left_Bundle_Branch_Block':'Left_Bundle_Branch_Block',
     'STEMI_Anterior_Ischemia':'STEMI_Anterior_Ischemia',
     'STEMI_Lateral_Ischemia':'STEMI_Lateral_Ischemia',
     'STEMI_Inferior_Ischemia':'STEMI_Inferior_Ischemia',
     'S':'Sinus_Rhythm',
     'Si':'Sinus_Rhythm',
     'Sin':'Sinus_Rhythm',
     'Sinu':'Sinus_Rhythm',
     'Sinus':'Sinus_Rhythm',
     'Sinus_':'Sinus_Rhythm',
     'Sinus_Arrhythmia':'Sinus_Arrhythmia',
     'Sinus_Bradycardia':'Sinus_Bradycardia',
     'Sinus_R':'Sinus_Rhythm',
     'Sinus_Rh':'Sinus_Rhythm',
     'Sinus_Rhy':'Sinus_Rhythm',
     'Sinus_Rhyth':'Sinus_Rhythm',
     'Sinus_Rhythm':'Sinus_Rhythm',
     'Sinus_Rhythm,Sinus_Tachycardia':'Sinus_Rhythm,Sinus_Tachycardia',
     'Sinus_T':'Sinus_Tachycardia',
     'Sinus_Tach':'Sinus_Tachycardia',
     'Sinus_Tachyc':'Sinus_Tachycardia',
     'Sinus_Tachycardi':'Sinus_Tachycardia',
     'Sinus_Tachycardia':'Sinus_Tachycardia',
     'Supravent':'Supraventricular_Tachycardia',
     'Supraventricular_Tachycardia':'Supraventricular_Tachycardia',
     'Ventricular_Fibrillation':'Ventricular_Fibrillation',
     'Ventricular_Tachycardia_(With_Pulse)':'Ventricular_Tachycardia',
     'Non_STEMI_Lateral_Ischemia':'Non_STEMI_Lateral_Ischemia'
}

pool = set(['Medical - Abdominal Pain',
            'Medical - Altered Mental Status',
            'Medical - Seizure',
            'Medical - Respiratory Distress/Asthma/COPD/Croup/Reactive Airway',
            'General - Behavioral/Patient Restraint',
            'Medical - Overdose/Poisoning - Opioid',
            'Medical - Diabetic - Hypoglycemia',
            'Medical - Chest Pain - Cardiac Suspected'])
            
FN2H = {
 '': 1,
 'aspirin': 2,
 'albuterol': 2,
 'bag valve mask ventilation': 4,
 'capnography': 2,
 'cardiac monitor': 2,
 'cpap': 4,
 'dexamethasone': 2,
 'dextrose': 4,
 'endotracheal tube':4,
 'fentanyl': 1,
 'glucagon': 3,
 'ipratropium': 2,
 'midazolam': 4,
 'morphine sulfate': 1,
 'narcan': 4,
 'nasopharyngeal airway': 4,
 'nitroglycerine': 3,
 'normal saline': 1,
 'ondansetron': 1,
 'oral glucose': 2,
 'oxygen': 3,
 'physical restraint': 3,
 'transport': 1,
 'suction the oropharynx': 3,
 'suction the nasopharynx': 3
}

FP2H = {
 '': 1,
 'aspirin': 2,
 'bag valve mask ventilation': 2,
 'cardiac monitor': 1,
 'cpap': 3,
 'dexamethasone': 2,
 'dextrose': 2,
 'midazolam': 4,
 'narcan': 2,
 'nitroglycerin': 2,
 'normal saline': 2,
 'ondansetron': 1,
 'oral glucose': 2,
 'oxygen': 1,
 'physical restraint': 2,
 'transport': 1,
 'fentanyl': 4,
 'endotracheal tube': 4
}
            
# start
# read labeled cases
#docu = '/Users/sileshu/Desktop/EMSdata/RAA_1000_test.xlsx'
#df = pd.read_excel(docu)
#narratives = df['Narrative']
#inters = df['Interventions']
    
#chunked_narratives = [textwrap.wrap(narrative, len(narrative) / 10) for narrative in narratives]


n = "this is paramedic Smith we are transporting a 36 year old male complaining of shortness of breath he also complains of non radiating sharp chest pain on expiration symptoms began 1 hour ago after exercising he attempted to use his inhaler without relief he does have a history of asthma this patient was found with obvious respiratory difficulty he is awake and appropriate on exam we find lung sounds with wheezes in Upper lobes bilaterally and quiet lower lobes bilaterally initial Vital Signs BP 162 over 94 pulse 120 for respirations 36 skin is warm and moistpulse oximetry 92% on 15 LPM by non-rebreather ECG sinus tach without a activate we've treated with high-flow oxygen followed by one dose of Proventil and atrovent by small volume nebulizer the patient has improved and now has respirations of 24 flight wheezes in all Fields Pulse oximetry 96% on 6 LPM by nebulizer we plan to begin a second dose of Proventil by nebulizer and transport to your ETA ETA is 7 minutes do you have any questions"

n = "McLaren Maycomb MedStar Alpha 117 Priority One traffic McLaren make them MedStar Alpha 117 Priority One For You 50252 year old male Chief complaint today is chest pain appears to be we did transmit you to 12 leads ID number on that is going to be 11:50 3 1 1 5 3 fi first 12 lead is a regular 12 lead showing stemi second 12 lead we sent you is actually right side it says v4r it does appear to be a right-sided mi at this time the patient is alert and oriented times 4 GCS is 15 Bibles for you blood pressure is 136 over 98 heart rate 73 sinus rhythm on the monitor SATs 98% nasal cannula on 4L we have 324 aspirin on board to Ivy's established going to administer a nitroglycerin patient states left side of chest pain radiating down to his left arm 7 or 8 or 10 now States on set about 1:30 this afternoon only history of hypertension if you require further we will be there in about 5 to 7 minutes patient does not have a cardiologist he only has a primary care doctor that he recently is no longer seen"

#chunked_narratives = [["hello how is it going", "hello world"],['a','b']]
chunked_narratives = [textwrap.wrap(n, len(n) / 10)]

print(chunked_narratives, "\n\n\n")


pred_int = []
# extract concept and calculate similarity
def pre_tick_handler(behaviour_tree):
    blackboard = Blackboard()
    blackboard.tick_num += 1

for i,chunks in enumerate(tqdm(chunked_narratives[:5])):
    # loop the documents
    temp_int = []
    blackboard = Blackboard()
    root = py_trees.composites.Sequence("Root_1")
    IG = be.InformationGathering()
    TC = be.TextCollection()
    V = be.Vectorize()
    PS = be.ProtocolSelector()
    root.add_children([TC,IG,V,PS,be.protocols])
    behaviour_tree = py_trees.trees.BehaviourTree(root)
    behaviour_tree.add_pre_tick_handler(pre_tick_handler)
    behaviour_tree.setup(15)
    for idx, item in enumerate(chunks):
        # loop the chunks
        blackboard.text = [item]
        behaviour_tree.tick_tock(
            sleep_ms=50,
            number_of_iterations=1,
            pre_tick_handler=None,
       post_tick_handler=None
        )
        res = []
        for key in blackboard.feedback:
            if blackboard.feedback[key] > 0.1:
                res.append((key, blackboard.feedback[key]))
        temp_int.append(res)
        displayRes()
        #print blackboard.candi[:3]
        #print res
    pred_int.append(temp_int)




