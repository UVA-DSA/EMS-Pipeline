# -*- coding: utf-8 -*-
from pymetamap import MetaMap
#mm = MetaMap.get_instance('/home/arif/Desktop/metamap/public_mm/bin/metamap20')
mm = MetaMap.get_instance('./public_mm/bin/metamap20')

#for abs path, add "/home/arif/Desktop/metamap/"

#dummy=["Ems and fire were dispatched to the behavioral clinic for a 51 year old female complaining of shortness of breath. Upon arriving on location, we found the female with Richmond fire department, the patient stated she had not been feeling well since this morning and had shortness of breath. She said that she had not been feeling well since this morning and it was describing a feeling of chest fullness with shortness of breath. She stated it felt similar to asthma and acid reflux, but worse than either of them, she said she had taken. She had attempted two treatments with her albuterol inhaler 30 minutes before our arrival, but did not gain any relief. She also complained of a headache and dizziness patient had no significant cardiac history and her only medical history with asthma, hypertension and behavioral disorders. The patient was alert and oriented times 4 answering all questions appropriately. She was able to speak in full sentences between breaths upon auscultation of the lungs only found wheezing in all quadrants. A 12-lead was performed and showed no ectopy and sinus rhythm. She was also placed on intitle monitoring after hearing the wheezing when she was given a duoneb treatment via nebulizer. At this time she decided that she wanted to be transported to VCU hospital to be checked out. We then assisted her from the wheelchair to the stretcher, secured her to the stretcher and transported without incident to VCU hospital care was transferred to nursing staff upon arrival."]
#dummy1=["We'Ve initiated an IV given a saline bolus and 4 mg of ondansetron and with accompanied Improvement in patient condition."]

def metamapExtract(sentenceList):
    conceptNumList=[]
    UMLSconceptList = []
    
    for i in range(len(sentenceList)):
        conceptNumList.append(i+1)
        
    concepts,error = mm.extract_concepts(sentenceList, conceptNumList)
    
    for concept in concepts:

        if (float(concept[2])>09.99):
            UMLSconceptList.append(concept)
            
# =============================================================================
#             
#             if ((("fndg" in concept[5]))):
#                 print("^^^^^^^^^^^^^^^^^^^")
#                 print(concept)
#                 print("^^^^^^^^^^^^^^^^^^^")
# =============================================================================
            
# =============================================================================
#     for conceptGT10 in UMLSconceptList:
#         print(conceptGT10[2])
#         print(conceptGT10[6][-2])
#         print("##########")
# =============================================================================
# =============================================================================
#     for i in range(len(UMLSconceptList)): 
#         print("*************")
#         print(UMLSconceptList[i])
#         #print(UMLSconceptList[i][2])
#         print("*************")
# =============================================================================
    
    return UMLSconceptList

def metamapExtractt(sentenceList,threshold):
    conceptNumList=[]
    UMLSconceptList = []
    
    for i in range(len(sentenceList)):
        conceptNumList.append(i+1)
        
    concepts,error = mm.extract_concepts(sentenceList, conceptNumList)
    
    for concept in concepts:

        if (float(concept[2])>threshold):
            UMLSconceptList.append(concept)
            
# =============================================================================
#             
#             if ((("fndg" in concept[5]))):
#                 print("^^^^^^^^^^^^^^^^^^^")
#                 print(concept)
#                 print("^^^^^^^^^^^^^^^^^^^")
# =============================================================================
            
# =============================================================================
#     for conceptGT10 in UMLSconceptList:
#         print(conceptGT10[2])
#         print(conceptGT10[6][-2])
#         print("##########")
# =============================================================================
# =============================================================================
#     for i in range(len(UMLSconceptList)): 
#         print("*************")
#         print(UMLSconceptList[i])
#         #print(UMLSconceptList[i][2])
#         print("*************")
# =============================================================================
    
    return UMLSconceptList

#metamapExtract(dummy)
#metamapExtractt(dummy1,2.99)
    
