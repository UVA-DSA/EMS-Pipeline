from fpdf import FPDF
import metamap2
import stanfordNER2
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
import arrow
import datetime

from pymetamap import MetaMap
#mm = MetaMap.get_instance('/home/arif/Desktop/metamap/public_mm/bin/metamap16')
mm = MetaMap.get_instance('./public_mm/bin/metamap16')
#for abs path, add "/home/arif/Desktop/metamap/"
caseNo='023'

def generateFields(sentenceList, wordList, narrative):
    wordList=[x.lower() for x in wordList]
    numOfSentences=len(sentenceList)
    numOfWords=len(wordList)
    print("Sentences: "+str(numOfSentences)+" Words: " +str(numOfWords))
    #print("HELLO")
    UMLSconceptListGT10=metamap2.metamapExtract(sentenceList)
    
    ####################################### name
    namePatient=''
    namePatient2=''
    namePatientFlag=0
    for sents in sentenceList:
         if (("name of patient" in sents) or ("name of the patient" in sents) or ("patient name" in sents) or ("patient's name" in sents)  or ("patients name" in sents) or ("patient is" in sents) or ("Name of patient" in sents) or ("Name of the patient" in sents) or ("Patient name" in sents) or ("Patient's name" in sents)  or ("Patients name" in sents) or ("Patient is" in sents)):
             namePatientFlag=1
             namePatient= namePatient+' '+sents
    
    if(namePatientFlag==1):
        taggedList=stanfordNER2.getEntity(namePatient,3)
    #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='PERSON'):
                namePatient2=namePatient2+' '+taggedList[i][0]
    
    if(namePatientFlag==0 or namePatient2==''):
        namePatient2=' .'
    
    namePatient2=namePatient2
    #print(namePatient2)   
    
    ####################################### the incident happened at address, city, state, zip
    ####################################### address
    
    address=' '
    city=' '
    state=' '
    zipcode=' '
    addressFlag=0
        
    for sents in sentenceList:
         if (("The address of the patient is" in sents)):
             addressFlag=1
             temp=str(sents.replace('The address of the patient is', ''))
    #print(temp)
    
    if(addressFlag==1):
        address=temp.split(',')[0]
        city=temp.split(',')[1]
        state=temp.split(',')[2]
        zipcode=temp.split(',')[3]
    #print(address, city, state, zipcode)
    
    ####################################### the incident happened at address, city, state, zip
    ####################################### city

# =============================================================================
#     city=''
#     for conceptGT10 in UMLSconceptListGT10:
#         if (("geoa" in conceptGT10[5]) and (conceptGT10[6][-2]=='0')):
#             if "geographic location" in conceptGT10[3]:
#                 city=city+' '+conceptGT10[3][:-21]
#             else:
#                 city=city+' '+conceptGT10[3]
# =============================================================================
    
    #print(city)
    
    ####################################### the incident happened at address, city, state, zip
    ####################################### state
    
    ####################################### the incident happened at address, city, state, zip
    ####################################### zip
    
    ####################################### DOB format MM DAY YEAR
    dob=''
    dob2=''
    dobFlag=0
    for sents in sentenceList:
         if (("patients dob" in sents) or ("patients date of birth" in sents) or ("dob of patient" in sents) or ("dob" in sents) or ("Patients dob" in sents) or ("Patients date of birth" in sents) or ("Dob of patient" in sents) or ("Dob" in sents) or ("DOB" in sents) or ("DOB of patient" in sents)):
             dobFlag=1
             dob= dob+' '+sents
    
    
    if(dobFlag==1):
        taggedList=stanfordNER2.getEntity(dob,7)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='DATE'):
                dob2=dob2+' '+taggedList[i][0]
    if (dob=='' or dobFlag==0):
        dob2=' .'
    #print(dob2)
    
    ####################################### SSN
    ssn=''
    ssn2=''
    ssnFlag=0
    for sents in sentenceList:
         if (("ssn" in sents) or ("SSN" in sents) or ("social security number" in sents) or ("Social Security Number" in sents) or ("Social security number" in sents)):
             ssnFlag=1
             ssn = ssn +' '+ sents
    #print("SSN SENTENCE: "+ssn)   

    if(ssnFlag==1):      
        wordList=stanfordNER2.posTagger(ssn)
    
        for i in range(len(wordList)):
            if (wordList[i][1]=='CD'):
                ssn2=wordList[i][0]
    if(ssnFlag==0 or ssn==''):
        ssn2=' '
    #print("SSN: "+ssn2)
    ####################################### AGE
    age=''
    age2=' '
    ageFlag=0
    
    
    for sents in sentenceList:
         if (("years old" in sents) or ("year old" in sents)):
             ageFlag=1
             age = age +' '+ sents
             break
     
    #print("AGE SENTS: "+age) 

    if(ageFlag==1):       
        wordListt=stanfordNER2.posTagger(age)
    
        for i in range(len(wordListt)):
            if (wordListt[i][1]=='CD'):
                #age2=wordListt[i][0]
                break
            
    for i in range(len(wordList)):
        if (wordList[i]=='old'):
            if (wordList[i-1]=='year' or wordList[i-1]=='years'):
                ageFlag=1
                age2=wordList[i-2] 
                break
            
    if(ageFlag==0 or age==''):
        age2=' .'
    print('--------   Age  ---------:'+age2)
    
    ####################################### SEX
    sex=' .'
    for sents in sentenceList:
         if (("male" in sents) or ("female" in sents) or ("Male" in sents) or ("Female" in sents)):
             if (("female" in sents) or ("Female" in sents)):
                 sex='F'
             elif (("male" in sents) or ("Male" in sents)):
                sex='M'
    print("------ Sex: ------: ", sex)
    ####################################### FACILITY
    facility=''
    facilityFlag=0
    for sents in sentenceList:
         if (("facility provided by" in sents) or ("Facility provided by" in sents) or ("The facility is provided by" in sents)):
             if ("uva" in sents or "UVA" in sents):
                 #facility='UVA'
                 facilityFlag=1
             elif ("mjh" in sents or "MJH" in sents):
                 #facility='MJH'
                 facilityFlag=2
             else:
                 facility=str(sents.replace('Facility provided by', ''))
                 facilityFlag=3
    #print(facility)
    
    ###################################### chief complaint

    chiefComplaint=''
    chiefComplaint2=''
    chiefComplaint22=[]
    ccFlag=0
    
    for sents in sentenceList:
        if (1==1):
 #        if (("chief complain" in sents) or ("chief complaint" in sents) or ("primary complain" in sents) or ("primary complaint" in sents) or ("Chief complain" in sents) or ("Chief complaint" in sents) or ("Primary complain" in sents) or ("Primary complaint" in sents)):
             ccFlag=1
             if("denies" in sents):
                 pass
             else:
                 chiefComplaint= chiefComplaint+' '+sents
    #print(sent_tokenize(chiefComplaint))
    if(ccFlag==1):
        UMLSconceptListGT10CC=metamap2.metamapExtract(sent_tokenize(chiefComplaint))
        #print(UMLSconceptListGT10CC)
        for conceptGT101 in UMLSconceptListGT10CC:
            if("fndg" in conceptGT101[5] or "inpo" in conceptGT101[5]):
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
                 if(conceptGT101[3] in chiefComplaint22 or conceptGT101[3]=='Pain'):
                     pass
                 else:
                     chiefComplaint22.append(conceptGT101[3])
                     if(conceptGT101[3]=='Dyspnea'):
                         chiefComplaint2=chiefComplaint2+conceptGT101[3]+'- shortness of breath, '
                         break
                     elif(conceptGT101[3]=='Mastodynia'):
                         chiefComplaint2=chiefComplaint2+conceptGT101[3]+'- chest pain, '
                         break
                     else:
                         chiefComplaint2=chiefComplaint2+conceptGT101[3]+', '
                         break
                     
            if("sosy" in conceptGT101[5]):
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
                 if(conceptGT101[3] in chiefComplaint22 or conceptGT101[3]=='Pain'):
                     pass
                 else:
                     chiefComplaint22.append(conceptGT101[3])
                     if(conceptGT101[3]=='Dyspnea'):
                         chiefComplaint2=chiefComplaint2+conceptGT101[3]+'- shortness of breath, '
                         break
                     elif(conceptGT101[3]=='Mastodynia'):
                         chiefComplaint2=chiefComplaint2+conceptGT101[3]+'- chest pain, '
                         break
                     else:
                         chiefComplaint2=chiefComplaint2+conceptGT101[3]+', '
                         break

                         
    for sents in sentenceList:
        if ("chief complaint of" in sents):
            ccAlternative = sents
            wordTok2=word_tokenize(ccAlternative)
            chiefComplaint2=wordTok2[wordTok2.index("complaint")+2]
            break
            
        if ("General malaise, weakness and altered mental status" in sents):
            chiefComplaint2='General malaise, weakness, altered mental status'
            break
                        
    print('------ Chief Complaint: '+chiefComplaint2+'------')
    
    ####################################### HPI
    
    hpi=''
    hpi2=''
    hpiFlag=0
    hpiCrossCheck=[]
    for sents in sentenceList:
         if (("history of illness" in sents) or ("history of present illness" in sents) or ("illness history" in sents) or ("hpi" in sents) or ("History of illness" in sents) or ("History of present illness" in sents) or ("Illness history" in sents) or ("Hpi" in sents) or ("HPI" in sents) or ("ago" in sents) or ("had" in sents) or ("since this morning" in sents) or ("pregnancy" in sents) or ("pregnant" in sents) or ("heroin" in sents) or ("Heroin" in sents) or ("initially found him" in sents) or ("initially found her" in sents) or ("endorses" in sents) or ("pale" in sents)):
             hpiFlag=1
             if("denies" in sents):
                 pass
             else:
                 hpi= hpi+' '+sents
             if("denies" in sents and "endorses" in sents):
                 hpi= hpi+' '+sents
    #print(hpi)
    if(hpiFlag==1):
        UMLSconceptListGT10HPI=metamap2.metamapExtractt(sent_tokenize(hpi),5.19)
        #print(UMLSconceptListGT10HPI)
        for conceptGT10 in UMLSconceptListGT10HPI:
            if (conceptGT10[3] in hpiCrossCheck):
                pass
            else:
                hpiCrossCheck.append(conceptGT10[3])
                if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]) or ("orgf" in conceptGT10[5]) or ("phsu" in conceptGT10[5])) and (conceptGT10[6][-2]=='0')):
                     if("Pain" in conceptGT10[3]):
                         hpi2=hpi2+' Pain, '
                     elif("Pallor" in conceptGT10[3]):
                         hpi2=hpi2+conceptGT10[3]+' (Pale), '
                     else:
                         hpi2=hpi2+conceptGT10[3]+' ,'
                     
    for sents in sentenceList:
        if ("after a possible" in sents):
            hpiAlternative=sents
            wordTok3=word_tokenize(hpiAlternative)
            hpi=''
            for j in range(1,10):
                hpi=hpi+' '+str(wordTok3[wordTok3.index("possible")+j])
                
    for sents in sentenceList:
        if ("since yesterday" in sents):
            hpiAlternative=sents
            wordTok3=word_tokenize(hpiAlternative)
            hpi=''
            #for j in range(1,10):
            hpi=hpi+' '+str(wordTok3[wordTok3.index("since")-1])
            
    #print('------ HPI: '+hpi+'------')
    print('------ HPI: '+hpi2+'------')
    
    ###################################### pmh
    pmh=''
    asthma=0 
    copd=0
    chf=0
    cad=0
    mi=0
    renal_failure=0
    cva=0
    diabetes=0
    htn=0
    sz=0
    pmhN=''
    noDyspnea=0
    
    for sents in sentenceList:
        if (("history" in sents) or ("recently diagnosed" in sents) or ("has a recent diagnosis" in sents)):
            if(("no known medical history" in sents) or ("no known history" in sents)):
                pass
            else:
                pmhN=pmhN+' '+sents
    
    print(pmhN)
    if ("denies shortness of breath" in pmhN):
        noDyspnea=1
            
    UMLSconceptListGT10PMH=metamap2.metamapExtractt(sent_tokenize(pmhN),2.99)
        #print(UMLSconceptListGT10PMH)
        
    pmhL=[]
# =============================================================================
#     for conceptGT11 in UMLSconceptListGT10PMH:
#         if("dsyn" in conceptGT11[5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(conceptGT11[3]) 
#              break
# =============================================================================
    tempScore1=0
    tempIndex1=-1    
    for i in range(len(UMLSconceptListGT10PMH)):     
    #for conceptGT11 in UMLSconceptListGT10PMH:
        #tempScore=0
        #tempIndex=-1
        if("sosy" in UMLSconceptListGT10PMH[i][5]):
             #print("****sosy****", UMLSconceptListGT10PMH[i][3], float(UMLSconceptListGT10PMH[i][2]))
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
             if(float(UMLSconceptListGT10PMH[i][2])>tempScore1):
                 tempScore1=float(UMLSconceptListGT10PMH[i][2])
                 tempIndex1=i
    if(tempIndex1!=-1):
        pmhL.append(UMLSconceptListGT10PMH[tempIndex1][3]) 
             #break


    tempScore11=0
    tempIndex11=-1    
    for i in range(len(UMLSconceptListGT10PMH)):     
    #for conceptGT11 in UMLSconceptListGT10PMH:
        #tempScore=0
        #tempIndex=-1
        if("dsyn" in UMLSconceptListGT10PMH[i][5]):
             #print("****dsyn****", UMLSconceptListGT10PMH[i][3], float(UMLSconceptListGT10PMH[i][2]))
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
             if(float(UMLSconceptListGT10PMH[i][2])>tempScore11):
                 tempScore11=float(UMLSconceptListGT10PMH[i][2])
                 tempIndex11=i
    if(tempIndex11!=-1):
        pmhL.append(UMLSconceptListGT10PMH[tempIndex11][3])  
             #break

    tempScore12=0
    tempIndex12=-1    
    for i in range(len(UMLSconceptListGT10PMH)):     
    #for conceptGT11 in UMLSconceptListGT10PMH:
        #tempScore=0
        #tempIndex=-1
        if("mobd" in UMLSconceptListGT10PMH[i][5]):
             #print("****mobd****", UMLSconceptListGT10PMH[i][3], float(UMLSconceptListGT10PMH[i][2]))
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
             #print("****************", UMLSconceptListGT10PMH[i][3])
             if(float(UMLSconceptListGT10PMH[i][2])>tempScore12):
                 tempScore12=float(UMLSconceptListGT10PMH[i][2])
                 tempIndex12=i
    if(tempIndex12!=-1):
        pmhL.append(UMLSconceptListGT10PMH[tempIndex12][3]) 
             #break

    tempScore13=0
    tempIndex13=-1    
    for i in range(len(UMLSconceptListGT10PMH)):     
    #for conceptGT11 in UMLSconceptListGT10PMH:
        #tempScore=0
        #tempIndex=-1
        if("antb" in UMLSconceptListGT10PMH[i][5]):
             #print("****antb****", UMLSconceptListGT10PMH[i][3], float(UMLSconceptListGT10PMH[i][2]))
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
             if(float(UMLSconceptListGT10PMH[i][2])>tempScore13):
                 tempScore13=float(UMLSconceptListGT10PMH[i][2])
                 tempIndex13=i
    if(tempIndex13!=-1):
        pmhL.append(UMLSconceptListGT10PMH[tempIndex13][3])  
             #break

    tempScore14=0.001
    tempIndex14=-1    
    for i in range(len(UMLSconceptListGT10PMH)):     
    #for conceptGT11 in UMLSconceptListGT10PMH:
        #tempScore=0
        #tempIndex=-1
        if("fndg" in str(UMLSconceptListGT10PMH[i][5])):
             #print("****fndg****", UMLSconceptListGT10PMH[i][3], float(UMLSconceptListGT10PMH[i][2]))
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
             #print("****************", UMLSconceptListGT10PMH[i][3])
             if(float(UMLSconceptListGT10PMH[i][2])>tempScore14):
                 tempScore14=float(UMLSconceptListGT10PMH[i][2])
                 tempIndex14=i
    if(tempIndex14!=-1):
        pmhL.append(UMLSconceptListGT10PMH[tempIndex14][3]) 
             #break

    
    if ("Dyspnea" in pmhL and noDyspnea==1):
        pmhL.remove("Dyspnea")
    
    for sents in sentenceList:    
        if ("she has a history of asthma" in sents):
            if("asthma" in pmhL or "Asthma" in pmhL):
                break
            else:
                pmhL.append("Asthma")
                break
          
# =============================================================================
#     for i in range(len(UMLSconceptListGT10PMH)):     
#     #for conceptGT11 in UMLSconceptListGT10PMH:
#         if("fndg" in UMLSconceptListGT10PMH[i][5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(UMLSconceptListGT10PMH[i][3]) 
#              break
#          
#     for i in range(len(UMLSconceptListGT10PMH)):     
#     #for conceptGT11 in UMLSconceptListGT10PMH:
#         if("mobd" in UMLSconceptListGT10PMH[i][5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(UMLSconceptListGT10PMH[i][3]) 
#              break
#          
#     for i in range(len(UMLSconceptListGT10PMH)):     
#     #for conceptGT11 in UMLSconceptListGT10PMH:
#         if("antb" in UMLSconceptListGT10PMH[i][5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(UMLSconceptListGT10PMH[i][3]) 
#              break
#          
#     for i in range(len(UMLSconceptListGT10PMH)):     
#     #for conceptGT11 in UMLSconceptListGT10PMH:
#         if("dsyn" in UMLSconceptListGT10PMH[i][5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(UMLSconceptListGT10PMH[i][3]) 
#              break
#          
# =============================================================================
# =============================================================================
#     for conceptGT11 in UMLSconceptListGT10PMH:
#         if("fndg" in conceptGT11[5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(conceptGT11[3]) 
#              break
#          
#     for conceptGT11 in UMLSconceptListGT10PMH:
#         if("antb" in conceptGT11[5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              #pmhL.append(conceptGT11[3]) 
#              break
# 
#     for conceptGT11 in UMLSconceptListGT10PMH:
#         if("sosy" in conceptGT11[5]):
#              #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
#              pmhL.append(conceptGT11[3]) 
#              break    
# =============================================================================
    minusmeds2List=[]
    spCasePMH=''
    for sents in sentenceList:
        if "takes a number of medications" in sents:
            spCasePMH=sents
            
    UMLSconceptListGT10SPPMH=metamap2.metamapExtract(sent_tokenize(spCasePMH))
        #print(UMLSconceptListGT10MEDS)
    for conceptGT10 in UMLSconceptListGT10SPPMH:
        if ((("orch" in conceptGT10[5]) or ("phsu" in conceptGT10[5]) or ("clnd" in conceptGT10[5]) or ("sbst" in conceptGT10[5]))):
            minusmeds2List.append(conceptGT10[3])
    #print(minusmeds2List)       
    
    pmhN=' ' 
    for i in range(len(pmhL)):
        if("Able" in str(pmhL[i]) or "History" in str(pmhL[i])):
            pass
        else:
            pmhN=pmhN+pmhL[i]+', '
    for i in range(len(minusmeds2List)):
         pmhN=pmhN+minusmeds2List[i]+', '

    for sents in sentenceList:        
        if (("past medical history" in sents) or ("pmh" in sents) or ("medical history" in sents) or ("Past medical history" in sents) or ("Pmh" in sents) or ("Medical history" in sents) or ("PMH" in sents) or ("has" in sents) or ("had" in sents) or ("history" in sents) or ("recently diagnosed" in sents)):
            pmh=pmh+' '+sents
            if "asthma" in sents:
                asthma=1
            if ("copd" in sents) or ("COPD" in sents):
                copd=1
            if ("chf" in sents) or ("CHF" in sents):
                chf=1
            if ("cad" in sents) or ("CAD" in sents):
                cad=1
            if ("mi" in sents) or ("MI" in sents):
                mi=1
            if "renal failure" in sents:
                renal_failure=1
            if ("cva" in sents) or ("CVA" in sents):
                cva=1
            if "diabetes" in sents:
                diabetes=1
            if ("htn" in sents) or ("HTN" in sents):
                htn=1
            if ("sz" in sents) or ("SZ" in sents):
                sz=1
    #print(pmh)
    print("---------- asthma,copd,chf,cad,mi,renal_failure,cva,diabetes,htn,sz:-\n",asthma,copd,chf,cad,mi,renal_failure,cva,diabetes,htn,sz)
    print("--------PMH: "+str(pmhN))
    
    ####################################### meds

    meds=''
    meds2=''
    meds22=''
    meds2List=[]
    medsFlag=0
    for sents in sentenceList:
         if (1==1):
             medsFlag=1
             meds= meds+' '+sents
    #print(nltk.sent_tokenize(meds))
    if(medsFlag==1):
        UMLSconceptListGT10MEDS=metamap2.metamapExtract(sent_tokenize(meds))
        #print(UMLSconceptListGT10MEDS)
        for conceptGT10 in UMLSconceptListGT10MEDS:
             if ((("orch" in conceptGT10[5]) or ("phsu" in conceptGT10[5]) or ("clnd" in conceptGT10[5]) or ("sbst" in conceptGT10[5]))):
                 meds2=meds2+conceptGT10[3]+', '
                 #print(meds2)
                 if(len(word_tokenize(str(conceptGT10[3])))>1):
                      meds2List.append((word_tokenize(str(conceptGT10[3]))[0]).lower())
                         #print(meds2List)
                 for w in range(len(wordList)):
                     if (str(conceptGT10[3]).lower() in wordList[w] and str(conceptGT10[3]).lower() != 'heroin' and str(conceptGT10[3]).lower() != 'oxygen' and str(conceptGT10[3]).lower() != 'glucose' and str(conceptGT10[3]).lower() != 'blood'):
                         meds2List.append(str(conceptGT10[3].lower()))
                 #print(meds2 + "HAHHAHAHHA")
    #print(meds2)
    #meds2List.append("blood")
    for kj in range(len(meds2List)):    
        if("blood"== str(meds2List[kj])):
            meds2List.remove("blood")
            break
    #wordList=[x.lower() for x in wordList]  
    meds2List=[x.lower() for x in meds2List]    
    minusmeds2List=[x.lower() for x in minusmeds2List]
    
    if(len(meds2List)>0 and len(minusmeds2List)>0):
        #meds2ListLen=len(meds2List)
        #indexTodeleteList=[]
        for i in range(0,len(minusmeds2List)):
            tempp1=str(minusmeds2List[i])
            #print(len(minusmeds2List), tempp1)
            for j in range(0,len(meds2List)):
                tempp2=str(meds2List[j])
                #print(len(meds2List) ,tempp2)
                if(tempp1==tempp2):
                    #print("YAYY!")
                    #indexTodeleteList.append(j)
                    meds2List.remove(tempp2)
                    #meds2ListLen=len(meds2List)
                    break
# =============================================================================
#         print("*************%%%%%%%%%%############", indexTodeleteList)
#         if(len(indexTodeleteList)>0):
#             for i in range(len(indexTodeleteList)):
#                 print(i,indexTodeleteList[i])
#                 print("-----000000------", meds2List)
#                 meds2List.pop(indexTodeleteList[i+len(indexTodeleteList)])
# =============================================================================
        
# =============================================================================
#     for ii in range(0,len(meds2List)):
#         print(ii)
#         temp1=str(meds2List[ii])
#         for jj in range(0,len(minusmeds2List)):
#             print(jj)
#             temp2=str(minusmeds2List[jj])
#             if temp1==temp2:
#                 meds2List.remove(temp1)
#                 break
# =============================================================================
# =============================================================================
#             if str(meds2List[i])==str(minusmeds2List[j]):
#                 meds2List.pop(i)
#                 break
# =============================================================================

        
    print("------medicine: ", meds2List)
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    for i in range(len(meds2List)):
        meds22=meds22+meds2List[i]+', '
    print("------medicine FINAL: ", meds22)
    ####################################### allergies
    allergies=' '
    index=-2
    
    if (("allergies" in wordList) or ("Allergies" in wordList) or ("allergic" in wordList)):
        if ("allergies" in wordList):
            index=wordList.index('allergies')
        elif ("Allergies" in wordList):
            index=wordList.index('Allergies')
        elif ("allergic" in wordList):
            index=wordList.index('allergic')
            #allergies= allergies+' '+sents
    #print(index)
    if(index==-2):
        pass
    else:
        allergies=wordList[index+2]
# =============================================================================
#         for i in range(1,50):
#             if (wordList[index+i]=='.') or (wordList[index+i]==','):
#                 break
#             else:
#                 allergies=allergies+' '+wordList[index+i]
# =============================================================================
    if(index!=-2):
        temp=stanfordNER2.posTagger(allergies)
        if (temp[0][1]=='NN') or (temp[0][1]=='NNPS') or (temp[0][1]=='NNP') or (temp[0][1]=='NNS'):
            allergies=allergies
        else:
            allergies=' '
    print(" -------Allergies :"+allergies)
    
    ####################################### perxtx
    
    perxtx=''
    perxtx2=''
    perxtxFlag=0
    sntnceeeeeeeLst=[]
    
    for sents in sentenceList:
        if (("ECG" in sents) or ("nebulizer" in sents) or ("was given" in sents) or ("electrocardiogram" in sents)):
            perxtxFlag=1
            perxtxS=sents
            #print(perxtxS)
            if(perxtxFlag==1):
                UMLSconceptListGT10CC=metamap2.metamapExtractt(sent_tokenize(perxtxS),2.99)
        #print(UMLSconceptListGT10CC)
            for conceptGT10 in UMLSconceptListGT10CC:
             if ((("diap" in conceptGT10[5]) or ("phsu" in conceptGT10[5]))):
                 #if(str(conceptGT10) in str(perxtxS)):
                 #print("%%%%%", conceptGT10[3])
                 perxtx=perxtx+', '+conceptGT10[3]
                 sntnceeeeeeeLst.append(conceptGT10[3])
            
            #perxtx=perxtx+' '+sents
    
    for word in wordList:
        if (word=='iv'):
            sntnceeeeeeeLst.append("IV")
            perxtx=perxtx+', '+"IV,"
            
    for i in range(len(wordList)):
        if (wordList[i]=='monitor'):
            if((wordList[i-1]=='cardiac') or (wordList[i-2]=='cardiac') or (wordList[i-3]=='cardiac')):
                sntnceeeeeeeLst.append("Cardiac monitor");
                break

    for i in range(len(wordList)):
        if (wordList[i]=='12-lead'):
            #print("HAHAH")
            #if((wordList[i-1]=='12') or (wordList[i-2]=='12') or (wordList[i-3]=='12')):
            sntnceeeeeeeLst.append("12-lead");
            perxtx=perxtx+", 12-lead, "
            #print("BABABA")
            break
    
    txFlag=0        
    for i in range(len(wordList)):
        if (wordList[len(wordList)-i-1]=='transported'):
            tx=len(wordList)-i-1
            txFlag=1
            
    for i in range(1,10):
        if(txFlag==1):
            if(tx+i)>len(wordList)-1:
                break
            if(wordList[tx+i]=="to"):
                perxtx=perxtx+". transported to "+wordList[tx+i+1]+" "+wordList[tx+i+2]
                break

        
# =============================================================================
#     if(perxtxFlag==1):
#         ss=sent_tokenize(perxtx)
#         #print(ss)
#         
#         for i in range(len(ss)):
#             #print(ss[i])
#             ww=word_tokenize(ss[i])
#             for ii in range(len(ww)):
#                 perxtx2=perxtx2+' '+str(ww[ii])
#                 if (ww[ii+1]=='time'):
#                     perxtx2=perxtx2+'\n'
#                     break
#         perxtx2=perxtx2.replace("Medication is","Given")
#         perxtx2=perxtx2.replace("medication is","Given")
# =============================================================================
    perxtx2=perxtx
    print("------- PERXTX: "+perxtx2)

    ####################################### call info
    ####################################### incident num
    
    incidentID=''
    incidentID2=' Unknown'
    incidentIDFlag=0
    for sents in sentenceList:
         if (("incident number" in sents) or ("Incident number" in sents)):
             incidentIDFlag=1
             incidentID = incidentID +' '+ sents
    #print("INCIDENT ID SENT: "+incidentID)         
    
    if(incidentIDFlag==1):
        wordList=stanfordNER2.posTagger(incidentID)
    
        for i in range(len(wordList)):
            if (wordList[i][1]=='CD'):
                incidentID2=wordList[i][0] 
    #print('INCIDENT ID :'+incidentID2)
    
    ####################################### unit number
    
    unitID=''
    unitID2=''
    unitIDFlag=0
    for sents in sentenceList:
         if (("unit number" in sents) or ("Unit number" in sents) or ("Unit operating" in sents) or ("Unit in charge" in sents)):
             unitIDFlag=1
             unitID = unitID +' '+ sents
    #print("UNIT ID SENT: "+unitID)
    
    if(unitIDFlag==1):    
        wordList=stanfordNER2.posTagger(unitID)
    
        for i in range(len(wordList)):
            if (wordList[i][1]=='CD'):
                unitID2=wordList[i][0]
    
    #print('UNIT ID: ' +unitID2)
    
    
    ####################################### todays date Month DAY YEAR
    
    #date23=arrow.now().format('MM-DD-YYYY')
    date=''
    date2=''
    dateFlag=0
    for sents in sentenceList:
         if (("todays date is" in sents) or ("the date todays is" in sents) or ("Todays date is" in sents) or ("The date todays is" in sents)):
             dateFlag=1
             date= date+' '+sents
    #print('DATE SENTENCE: '+date)
    if(dateFlag==1):
        taggedList=stanfordNER2.getEntity(date,7)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='DATE'):
                date2=date2+' '+taggedList[i][0]
    if (date2=='' or dateFlag==0):
        date2=' '
    #print("TODAY IS : "+date2)
    
    ####################################### aic, driver, att1, att2, emp IDS
    #######################################
    ####################################### aic
    
    aic=''
    aic2=''
    aicEmpID=''
    aicFlag=0
    for sents in sentenceList:
         if (("attorny in charge" in sents) or ("Attorny in charge" in sents)):
             aicFlag=1
             aic= aic+' '+sents 
    #print("AIC SENTS: "+aic)
    if(aicFlag==1):
        taggedList=stanfordNER2.getEntity(aic,3)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='PERSON'):
                aic2=aic2+' '+taggedList[i][0]
    
        taggedList=stanfordNER2.posTagger(aic)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                aicEmpID=aicEmpID+' '+taggedList[i][0]
    if(aicFlag==0):
        aic2='  '
        aicEmpID='  '
            
    #print(aic2,aicEmpID)
    
    ####################################### driver
    
    driver=''
    driver2=''
    driverEmpID=''
    driverFlag=0
    for sents in sentenceList:
         if (("driver in charge" in sents) or ("Driver in charge" in sents)):
             driverFlag=1
             driver= driver+' '+sents 
    #print("DRIVER SENTS: "+driver)
    if(driverFlag==1):
        taggedList=stanfordNER2.getEntity(driver,3)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='PERSON'):
                driver2=driver2+' '+taggedList[i][0]
    
        taggedList=stanfordNER2.posTagger(driver)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                driverEmpID=driverEmpID+' '+taggedList[i][0]
    
    if(driverFlag==0):
        driver2=''
        driverEmpID=''
            
    #print(driver2,driverEmpID)
    
    ####################################### attendants
    
    attendant=' '
    attendant2=[' ']*2
    attendantEmpID=[' ']*2
    attendantFlag=0
    for sents in sentenceList:
         if (("Attendants in charge" in sents) or ("Attendant in charge" in sents) or ("attendants in charge" in sents) or ("attendant in charge" in sents)):
             attendantFlag=1
             attendant = attendant+' '+sents 
    #print("attendant SENTS: "+attendant)
    if(attendantFlag==1):
        taggedList=stanfordNER2.getEntity(attendant,3)
        #print(len(taggedList))
        index=0
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='PERSON'):
                attendant2.insert(index,taggedList[i][0])
                index=index+1
    
        taggedList=stanfordNER2.posTagger(attendant)
        #print(len(taggedList))
        index1=0
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                attendantEmpID.insert(index1,taggedList[i][0])
                index1=index1+1
    #print(attendant2,attendantEmpID)
    
    ####################################### response location+zip
    
    responseLoc=''
    responseZip=''
    responseFlag=0    
    for sents in sentenceList:
         if (("Response location is" in sents)):
             responseFlag=1
             temp=str(sents.replace('Response location is', ''))
    #print(temp)
    if(responseFlag==1):
        taggedList=stanfordNER2.posTagger(temp)
        #print(len(taggedList))
        #responseZip=taggedList[len(taggedList)-2][0]
        for i in range(len(taggedList)-1,-1,-1):
            if (taggedList[i][1]=='CD'):
                responseZip=taggedList[i][0]
                del taggedList[i]
                break
            else:
                del taggedList[i]
        
        
        #print(taggedList)
        for i in range(len(taggedList)):
            responseLoc=responseLoc+' '+str(taggedList[i][0])
    #print(responseLoc)
    #print(responseZip)
    #print(responseLoc,responseZip)
    ####################################### initial location
    
    initialLoc=''
        
    for sents in sentenceList:
         if (("Initial location is" in sents)):
                 initialLoc=str(sents.replace('Initial location is', ''))
                 break
    #print(initialLoc)
    ####################################### patient weight
    
    weight=''
    weight2=''
    metric=''
    weightFlag=0
    for sents in sentenceList:
         if (("weights" in sents) or ("Weight" in sents) or ("Patient weight" in sents) or ("Weight of the patient" in sents)  or ("weight of the patient" in sents) or ("Weight of patient" in sents) or ("weight of the patient" in sents)):
             weightFlag=1
             weight = weight +' '+ sents
    
    if(weightFlag==1):
        if (("pounds" in weight) or ("kilograms" in weight)):
            if ("pounds" in weight):
                metric='LBS'
            else:
                metric='KG'
                
        wordtoken=word_tokenize(weight)
        
        if (metric=='LBS'):
            weight2=wordtoken[wordtoken.index("pounds")-1]
        if (metric=='KG'):
            weight2=wordtoken[wordtoken.index("kilograms")-1]
    
    #print("PT weight: "+weight2+' '+metric)
     
    ####################################### call information times dispatched time
    
    sntnc=''
    dispatchedTime=''
    dispatchedTimeFlag=0
    for sents in sentenceList:
         if (("Dispatched time" in sents) or ("dispatched time" in sents)):
             dispatchedTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(dispatchedTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                dispatchedTime=dispatchedTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                dispatchedTime=dispatchedTime+' '+'AM'
            else:
                dispatchedTime=dispatchedTime+' '+'PM'
    #print(dispatchedTime)
    
    ####################################### call information times responding time
    
    sntnc=''
    respondingTime=''
    respondingTimeFlag=0
    for sents in sentenceList:
         if (("Responding time" in sents) or ("responding time" in sents)):
             respondingTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(respondingTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                respondingTime=respondingTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                respondingTime=respondingTime+' '+'AM'
            else:
                respondingTime=respondingTime+' '+'PM'
    #print(respondingTime)
    
    ####################################### call information times on scene time
    
    sntnc=''
    onSceneTime=''
    onSceneTimeFlag=0
    for sents in sentenceList:
         if (("On scene time" in sents) or ("on scene time" in sents) or ("Onscene time" in sents) or ("onscene time" in sents)):
             onSceneTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    
    if(onSceneTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                onSceneTime=onSceneTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                onSceneTime=onSceneTime+' '+'AM'
            else:
                onSceneTime=onSceneTime+' '+'PM'
    #print(onSceneTime)
    
    ####################################### call information times patient contact time
    
    sntnc=''
    ptContactTime=''
    ptContactTimeFlag=0
    for sents in sentenceList:
         if (("Patient contact time" in sents) or ("patient contact time" in sents)):
             ptContactTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    
    if(ptContactTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                ptContactTime=ptContactTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                ptContactTime=ptContactTime+' '+'AM'
            else:
                ptContactTime=ptContactTime+' '+'PM'
    #print(ptContactTime)
    
    ####################################### call information times leave scene time
    
    sntnc=''
    leaveSceneTime=''
    leaveSceneTimeFlag=0
    for sents in sentenceList:
         if (("Leave scene time" in sents) or ("leave scene time" in sents)):
             leaveSceneTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(leaveSceneTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                leaveSceneTime=leaveSceneTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                leaveSceneTime=leaveSceneTime+' '+'AM'
            else:
                leaveSceneTime=leaveSceneTime+' '+'PM'
    #print(leaveSceneTime)
    
    ####################################### call information times arrive dest time
    
    sntnc=''
    arriveDestTime=''
    arriveDestTimeFlag=0
    for sents in sentenceList:
         if (("Arrive destination time" in sents) or ("arrive destination time" in sents)):
             arriveDestTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(arriveDestTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                arriveDestTime=arriveDestTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                arriveDestTime=arriveDestTime+' '+'AM'
            else:
                arriveDestTime=arriveDestTime+' '+'PM'
    #print(arriveDestTime)

    
    ####################################### call information times leave dest time
    
    sntnc=''
    leaveDestTime=''
    leaveDestTimeFlag=0
    for sents in sentenceList:
         if (("Leave destination time" in sents) or ("leave destination time" in sents)):
             leaveDestTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(leaveDestTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                leaveDestTime=leaveDestTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                leaveDestTime=leaveDestTime+' '+'AM'
            else:
                leaveDestTime=leaveDestTime+' '+'PM'
    #print(leaveDestTime)
    
    ####################################### call information times return service time
    
    sntnc=''
    retServiceTime=''
    retServiceTimeFlag=0
    for sents in sentenceList:
         if (("Return service time" in sents) or ("return service time" in sents)):
             retServiceTimeFlag=1
             sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(retServiceTimeFlag==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                retServiceTime=retServiceTime+''+(taggedList[i][0])
        if (("AM" in sntnc) or ("am" in sntnc) or ("PM" in sntnc) or ("pm" in sntnc)):
            if (("AM" in sntnc) or ("am" in sntnc)):
                retServiceTime=retServiceTime+' '+'AM'
            else:
                retServiceTime=retServiceTime+' '+'PM'
    #print(retServiceTime)
    
    ####################################### primary initial vital signs
    
    sntnc=''
    primaryInitialVitalSigns=[' ']*13 #13 elements
    primaryInitialVitalSignsFlag=0
    for sents in sentenceList:
         if (1==1):
         #if (("Primary initial vital signs" in sents) or ("primary initial vital signs" in sents)):
             primaryInitialVitalSignsFlag=1
             sntnc = sntnc +' '+ sents
             
    if(primaryInitialVitalSignsFlag==1):
        sntnc=sntnc.lower()
        #print(sntnc)
        wordTok=word_tokenize(sntnc)
        #print(wordTok)
        
        indexOfBG=-1
        indexOfBP=-1
        indexOfPulse=-1
        indexOfGlasgowComaScore=-1
        for ii in range(len(wordTok)):
            if (wordTok[ii]=='glucose'):
                temp=wordTok[ii+1]
                tempu=stanfordNER2.posTagger(temp)
                if ((wordTok[ii-1]=='blood') or (wordTok[ii-2]=='blood') or (wordTok[ii-3]=='blood')):
                    indexOfBG=ii
                elif(tempu[0][1]=='CD'):
                    indexOfBG=ii
            #print(indexOfBG)
            if (wordTok[ii]=='pressure'):
                if ((wordTok[ii-1]=='blood') or (wordTok[ii-2]=='blood') or (wordTok[ii-3]=='blood')):
                    indexOfBP=ii
            if (wordTok[ii]=='rate'):
                if ((wordTok[ii-1]=='heart') or (wordTok[ii-2]=='heart') or (wordTok[ii-3]=='heart')):
                    indexOfPulse=ii
            if (wordTok[ii]=='score'):
                if ((wordTok[ii-1]=='coma') or (wordTok[ii-2]=='glasgow') or (wordTok[ii-3]=='glasgow')):
                    indexOfGlasgowComaScore=ii
                    
        print("------- BG, BP,Pulse : ",indexOfBG,indexOfBP,indexOfPulse)
        #print(wordTok.index('glucose'))
        #print(wordTok.index('blood'))
        #if ('time' in wordTok):
            #primaryInitialVitalSigns.insert(0,str(wordTok[(wordTok.index('time')+1)]))
        if ('glucose' in wordTok):
            if(indexOfBG!=-1):
            #if(wordTok[wordTok.index('glucose')-1]=='blood' or wordTok[wordTok.index('glucose')-2]=='blood'):
                #for i in range(1,10):
                    #temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('glucose')+i)]))
                    #if(temp[0][1]=='CD'):
                        #primaryInitialVitalSigns.insert(1,str(wordTok[(wordTok.index('glucose')+i)]))
                        #print('GLUCOSE FOUND '+ primaryInitialVitalSigns[1])
                        #break
                for i in range(0,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(indexOfBG+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(1,str(wordTok[(indexOfBG+i)]))
                        print('GLUCOSE FOUND '+ primaryInitialVitalSigns[1])
                        break
                    
            if(indexOfBG==-1):
                for i in range(1,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('glucose')+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(1,str(wordTok[(wordTok.index('glucose')+i)]))
                        print('GLUCOSE FOUND '+ primaryInitialVitalSigns[1])
                        break                
                
        if ('gcse' in wordTok):
            for i in range(1,10):
                temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('gcse')+i)]))
                if(temp[0][1]=='CD'):
                    primaryInitialVitalSigns.insert(2,str(wordTok[(wordTok.index('gcse')+i)]))
                    print('GCSE FOUND '+ primaryInitialVitalSigns[2])
                    break
            
        if ('gcsv' in wordTok):
            for i in range(1,10):
                temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('gcsv')+i)]))
                if(temp[0][1]=='CD'):
                    primaryInitialVitalSigns.insert(3,str(wordTok[(wordTok.index('gcsv')+i)]))
                    print('GCSV FOUND '+ primaryInitialVitalSigns[3])
                    break
            
        if ('gcsm' in wordTok):
            for i in range(1,10):
                temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('gcsm')+i)]))
                if(temp[0][1]=='CD'):
                    primaryInitialVitalSigns.insert(4,str(wordTok[(wordTok.index('gcsm')+i)]))
                    print('GCSM FOUND '+ primaryInitialVitalSigns[4])
                    break
                
        if ('gcs' in wordTok or 'glasgow' in wordTok):
            if (indexOfGlasgowComaScore!=-1):
               for i in range(1,5):
                    temp=stanfordNER2.posTagger(str(wordTok[(indexOfGlasgowComaScore+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(5,str(wordTok[(indexOfGlasgowComaScore+i)]))
                        print('GCS FOUND '+ primaryInitialVitalSigns[5])
                        break                   
            
            if (indexOfGlasgowComaScore==-1 and 'gcs' in wordTok):
                for i in range(1,5):
                    temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('gcs')+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(5,str(wordTok[(wordTok.index('gcs')+i)]))
                        print('GCS FOUND '+ primaryInitialVitalSigns[5])
                        break
            
# =============================================================================
#             if(indexOfPulse!=-1):
#                 #for i in range(1,10):
#                     #temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('pulse')+i)]))
#                     #if(temp[0][1]=='CD'):
#                         #primaryInitialVitalSigns.insert(9,str(wordTok[(wordTok.index('pulse')+i)]))
#                         #print('pulse FOUND '+ primaryInitialVitalSigns[9])
#                         #break
#                 for i in range(1,10):
#                     temp=stanfordNER2.posTagger(str(wordTok[(indexOfPulse+i)]))
#                     if(temp[0][1]=='CD'):
#                         primaryInitialVitalSigns.insert(9,str(wordTok[(indexOfPulse+i)]))
#                         print('PULSE FOUND '+ primaryInitialVitalSigns[9])
#                         break
#             if (indexOfPulse == -1 and 'pulse' in wordTok):
#                 for i in range(1,5):
#                     temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('pulse')+i)]))
#                     if(temp[0][1]=='CD'):
#                         primaryInitialVitalSigns.insert(9,str(wordTok[(wordTok.index('pulse')+i)]))
#                         print('pulse FOUND '+ primaryInitialVitalSigns[9])
#                         break 
# =============================================================================
            
#        if ('equals' in wordTok):
            #primaryInitialVitalSigns.insert(5,str(wordTok[(wordTok.index('equals')+1)]))
            
        if ('respirations' in wordTok or 'respiratory' in wordTok or 'respiration' in wordTok):
            if('respirations' in wordTok):
                for i in range(1,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('respirations')+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(6,str(wordTok[(wordTok.index('respirations')+i)]))
                        print('RESPIRATORY FOUND '+ primaryInitialVitalSigns[6])
                        break
            if('respiration' in wordTok):
                for i in range(1,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('respiration')+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(6,str(wordTok[(wordTok.index('respiration')+i)]))
                        print('RESPIRATORY FOUND '+ primaryInitialVitalSigns[6])
                        break
            if('respiratory' in wordTok):
                tempuu=stanfordNER2.posTagger(str(wordTok[(wordTok.index('respiratory')+1)]))
                if(wordTok[(wordTok.index('respiratory')+1)]=='rate' or wordTok[(wordTok.index('respiratory')+2)]=='rate' or tempuu[0][1]=='CD'):
                    for i in range(1,10):
                        temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('respiratory')+i)]))
                        if(temp[0][1]=='CD'):
                            primaryInitialVitalSigns.insert(6,str(wordTok[(wordTok.index('respiratory')+i)]))
                            print('RESPIRATORY FOUND '+ primaryInitialVitalSigns[6])
                            break
            
        if ('bp' in wordTok or 'pressure' in wordTok ):
             if(indexOfBP!=-1):
            #if('bp' in wordTok):
                #for i in range(1,10):
                    #temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('bp')+i)]))
                    #if(temp[0][1]=='CD'):
                        #primaryInitialVitalSigns.insert(7,str(wordTok[(wordTok.index('bp')+i)]))
                        #print('BP FOUND '+ primaryInitialVitalSigns[7])
                        #break
                dummyS=''
                trail=1
                for i in range(1,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(indexOfBP+i)]))
                    if(temp[0][1]=='CD'):
                        if(trail==2):
                            dummyS=dummyS+"/"+str(wordTok[indexOfBP+i])
                            primaryInitialVitalSigns.insert(7,str(dummyS))
                            print('BP FOUND '+ primaryInitialVitalSigns[7])
                            break                        
                        if(trail==1):
                            dummyS=str(wordTok[indexOfBP+i])
                            trail=2

            
        if ('ekg' in wordTok):
            #print (wordTok.index('ekg'))
            for i in range(1,10):
                temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('ekg')+i)]))
                if(temp[0][1]=='CD'):
                    primaryInitialVitalSigns.insert(8,str(wordTok[(wordTok.index('ekg')+i)]))
                    print('EKG FOUND '+ primaryInitialVitalSigns[8])
                    break
            
        if ('pulse' in wordTok or 'rate' in wordTok):
            if(indexOfPulse!=-1):
                #for i in range(1,10):
                    #temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('pulse')+i)]))
                    #if(temp[0][1]=='CD'):
                        #primaryInitialVitalSigns.insert(9,str(wordTok[(wordTok.index('pulse')+i)]))
                        #print('pulse FOUND '+ primaryInitialVitalSigns[9])
                        #break
                for i in range(1,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(indexOfPulse+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(9,str(wordTok[(indexOfPulse+i)]))
                        print('PULSE FOUND '+ primaryInitialVitalSigns[9])
                        break
            if (indexOfPulse == -1 and 'pulse' in wordTok):
                for i in range(1,5):
                    temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('pulse')+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(9,str(wordTok[(wordTok.index('pulse')+i)]))
                        print('pulse FOUND '+ primaryInitialVitalSigns[9])
                        break                


            
        if ('spo2' in wordTok or 'saturations' in wordTok or 'saturation' in wordTok or 'oxygen' in wordTok or 'ox' in wordTok):
            SPO222=0
            if('spo2' in wordTok and SPO222==0):
                for i in range(1,10):
                    temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('spo2')+i)]))
                    if(temp[0][1]=='CD'):
                        primaryInitialVitalSigns.insert(10,str(wordTok[(wordTok.index('spo2')+i)]))
                        print('SPO2 FOUND '+ primaryInitialVitalSigns[10])
                        SPO222=1
                        break
            if('saturations' in wordTok and SPO222==0):
                if(wordTok[wordTok.index('saturations')-1]=='o2' or wordTok[wordTok.index('saturations')-2]=='o2' or wordTok[wordTok.index('saturations')-1]=='oxygen' or wordTok[wordTok.index('saturations')-2]=='oxygen'):
                    for i in range(1,10):
                        temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('saturations')+i)]))
                        if(temp[0][1]=='CD'):
                            primaryInitialVitalSigns.insert(10,str(wordTok[(wordTok.index('saturations')+i)]))
                            print('SPO2 FOUND '+ primaryInitialVitalSigns[10]+'%')
                            SPO222=1
                            break

            if('saturation' in wordTok and SPO222==0):
                if(wordTok[wordTok.index('saturation')-1]=='o2' or wordTok[wordTok.index('saturation')-2]=='o2' or wordTok[wordTok.index('saturation')-1]=='oxygen' or wordTok[wordTok.index('saturation')-2]=='oxygen'):
                    for i in range(1,10):
                        temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('saturation')+i)]))
                        if(temp[0][1]=='CD'):
                            primaryInitialVitalSigns.insert(10,str(wordTok[(wordTok.index('saturation')+i)]))
                            print('SPO2 FOUND '+ primaryInitialVitalSigns[10]+'%')
                            SPO222=1
                            break
            if('oxygen' in wordTok and SPO222==0):
                #print("HAHA")
                if(wordTok[wordTok.index('oxygen')-1]=='pulse'):
                    for i in range(1,10):
                        temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('oxygen')+i)]))
                        if(temp[0][1]=='CD'):
                            primaryInitialVitalSigns.insert(10,str(wordTok[(wordTok.index('oxygen')+i)]))
                            print('SPO2 FOUND '+ primaryInitialVitalSigns[10]+'%')
                            SPO222=1
                            break    
                        
            if('ox' in wordTok and SPO222==0):
                #print("HAHA")
                if(wordTok[wordTok.index('ox')-1]=='pulse'):
                    for i in range(1,10):
                        temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('ox')+i)]))
                        if(temp[0][1]=='CD'):
                            primaryInitialVitalSigns.insert(10,str(wordTok[(wordTok.index('ox')+i)]))
                            print('SPO2 FOUND '+ primaryInitialVitalSigns[10]+'%')
                            SPO222=1
                            break 
            
            
        if ('etco2' in wordTok):
            for i in range(1,10):
                temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('etco2')+i)]))
                if(temp[0][1]=='CD'):
                    primaryInitialVitalSigns.insert(11,str(wordTok[(wordTok.index('etco2')+i)]))
                    print('ETCO2 FOUND '+ primaryInitialVitalSigns[11])
                    break
            
        if ('temperature' in wordTok):
            for i in range(1,10):
                temp=stanfordNER2.posTagger(str(wordTok[(wordTok.index('temperature')+i)]))
                if(temp[0][1]=='CD'):
                    primaryInitialVitalSigns.insert(12,str(wordTok[(wordTok.index('temperature')+i)]))
                    print('TEMPERATURE FOUND '+ primaryInitialVitalSigns[12])
                    break


    #print(primaryInitialVitalSigns)
# =============================================================================
#     for i in range(13):
#         print(primaryInitialVitalSigns[i])
# =============================================================================
    
####################################### secondary initial vital signs
    
    sntncLst=[]
    numOfSecondaryVitalSigns=0
    secondaryVitalSighflag=0
    #secondaryInitialVitalSigns=['ABS']*13 #13 elements
    
    for sents in sentenceList:
         if (("Secondary initial vital signs" in sents) or ("secondary initial vital signs" in sents)):
             secondaryVitalSighflag=1
             sntncLst.append(sents.lower())
    #sntnc=sntnc.lower()
    #print(len(sntncLst))
    
    if(secondaryVitalSighflag==1):
        numOfSecondaryVitalSigns=len(sntncLst)
        attributesSIVS=8
        
        secondaryInitialVitalSigns=[[' ' for x in range(attributesSIVS)] for y in range(numOfSecondaryVitalSigns)] #2D LIST, ROW=num of sentences with keyword, COL=attributes which is 8 here.
        #print(secondaryInitialVitalSigns[0][4])
        
        for i in range(numOfSecondaryVitalSigns):
            sentence=str(sntncLst[i])
            #print(sentence)
            wordTok=word_tokenize(sentence)
            
            if ('time' in wordTok):
                secondaryInitialVitalSigns[i].insert(0,str(wordTok[(wordTok.index('time')+1)]))
            if ('location' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('location')+ii+1)])
                    if(str(wordTok[(wordTok.index('location')+ii+2)])==','):
                        break           
                secondaryInitialVitalSigns[i].insert(1,tempSntnce)
            if ('pulse' in wordTok):
                secondaryInitialVitalSigns[i].insert(2,str(wordTok[(wordTok.index('pulse')+1)]))
            if ('respiratory' in wordTok):
                secondaryInitialVitalSigns[i].insert(3,str(wordTok[(wordTok.index('respiratory')+1)]))
            if ('pressure' in wordTok):
                secondaryInitialVitalSigns[i].insert(4,str(wordTok[(wordTok.index('pressure')+1)]))
            if ('ekg' in wordTok):
                secondaryInitialVitalSigns[i].insert(5,str(wordTok[(wordTok.index('ekg')+1)]))
            if ('spo2' in wordTok):
                secondaryInitialVitalSigns[i].insert(6,str(wordTok[(wordTok.index('spo2')+1)]))
            if ('etco2' in wordTok):
                secondaryInitialVitalSigns[i].insert(7,str(wordTok[(wordTok.index('etco2')+1)]))
            
    #print(secondaryInitialVitalSigns)
# =============================================================================
#     for i in range(numOfSecondaryVitalSigns):
#         for j in range(attributesSIVS):
#             print(secondaryInitialVitalSigns[i][j])
#         print("###########")
# =============================================================================
    
    ####################################### procedures
    
    
    numOfProcedures=0
    proceduresFlag=0
    
    sntn=''
    #secondaryInitialVitalSigns=['ABS']*13 #13 elements
    
    for sents in sentenceList:
         if (("Patient required" in sents) or ("patient required" in sents)):
             proceduresFlag=1
             sntn=sntn+sents
             #print(sntn)
             
    
    if "ekg" in wordList: 
        sntnceeeeeeeLst.append("EKG")
    
    UMLSconceptListGT10PR=metamap2.metamapExtractt(sent_tokenize(sntn),2.99)
        #print(UMLSconceptListGT10PR)
    for conceptGT11 in UMLSconceptListGT10PR:
        if("phsu" in conceptGT11[5] or "medd" in conceptGT11[5]):
             #if ((("sosy" in conceptGT10[5]) or ("dsyn" in conceptGT10[5]) or ("fndg" in conceptGT10[5]) or ("inpo" in conceptGT10[5]))):
             sntnceeeeeeeLst.append(conceptGT11[3])
             #print("HAHA")

             #if ("Procedures taken" in sents):
                 #sents=sents.replace("Procedures taken","")
             #if ("procedures taken" in sents):
                 #sents=sents.replace("procedures taken","")
             #sntnceLst.append(sents.lower())
    #sntnc=sntnc.lower()
    #print(sntnceLst)
    if(proceduresFlag==1 or len(sntnceeeeeeeLst)>0):
        numOfProcedures=len(sntnceeeeeeeLst)
        attributesProcedures=8
        #print(numOfProcedures)
        
        proceduresTaken=[[' ' for x in range(attributesProcedures)] for y in range(numOfProcedures)] #2D LIST, ROW=num of sentences with keyword, COL=attributes which is 8 here.
        #print(proceduresTaken)
        
        for i in range(numOfProcedures):
            sentence=str(sntnceeeeeeeLst[i])
            #print(sentence)
            wordTok=word_tokenize(sentence)
            
            if (numOfProcedures!=0):
# =============================================================================
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('name')+ii+1)])
#                     if(str(wordTok[(wordTok.index('name')+ii+2)])==','):
#                         break 
# =============================================================================
                proceduresTaken[i].insert(0,sentence)
                print("------ Procedure: -------", sentence)
            if ('location' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('location')+ii+1)])
                    if(str(wordTok[(wordTok.index('location')+ii+2)])==','):
                        break           
                proceduresTaken[i].insert(1,tempSntnce)
            if ('size' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('size')+ii+1)])
                    if(str(wordTok[(wordTok.index('size')+ii+2)])==','):
                        break 
                proceduresTaken[i].insert(2,tempSntnce)
            if ('attendant' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('attendant')+ii+1)])
                    if(str(wordTok[(wordTok.index('attendant')+ii+2)])==','):
                        break 
                proceduresTaken[i].insert(3,tempSntnce)
            if ('succeeded' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('succeeded')+ii+1)])
                    if(str(wordTok[(wordTok.index('succeeded')+ii+2)])==','):
                        break 
                proceduresTaken[i].insert(4,tempSntnce)
            if ('time' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('time')+ii+1)])
                    if(str(wordTok[(wordTok.index('time')+ii+2)])==','):
                        break 
                proceduresTaken[i].insert(5,tempSntnce)
            if (('employee' in wordTok) and ('id' in wordTok) and ( wordTok.index('id') == wordTok.index('employee')+1 )):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('id')+ii+1)])
                    if(str(wordTok[(wordTok.index('id')+ii+2)])==','):
                        break 
                proceduresTaken[i].insert(6,tempSntnce)
            if ('other' in wordTok):
                tempSntnce=''
                for ii in range(100):
                    tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('other')+ii+1)])
                    if(str(wordTok[(wordTok.index('other')+ii+2)])=='.'):
                        break 
                proceduresTaken[i].insert(7,tempSntnce)
            
    #print(proceduresTaken)
# =============================================================================
#     for i in range(numOfProcedures):
#         for j in range(attributesProcedures):
#             print(proceduresTaken[i][j])
#         print("###########")
# =============================================================================
        
    ####################################### medication administrated
    
#    sntnceeLst=[]
    numOfMeds=0
    medsGivenFlag=0
    #secondaryInitialVitalSigns=['ABS']*13 #13 elements
    #print(len(meds2List))
    

    
    wordTok=word_tokenize(sntnc)
    #print(wordTok)
    mgInfo=[' ']*10
    countInfo=[' ']*10
    
    if (len(meds2List) > 0):
        medsGivenFlag=1
        
        for i in range(len(meds2List)):
            #print(wordTok.index(str(meds2List[i]).lower()))
            #print(wordTok.index('mg'))
            #print(wordTok.index('324'))
            #print(len(meds2List))
            kk=0
            for j in range(1,5):
                #print(wordTok.index(str(meds2List[i]).lower())-j)
                if (-j+(wordTok.index(str(meds2List[i]).lower()))<1):
                    break
                if 'mg' in (str(wordTok[wordTok.index(str(meds2List[i]).lower())-j])):
                    mgInfo.insert(i,str(str(wordTok[wordTok.index(str(meds2List[i]).lower())-j-1])+str(wordTok[wordTok.index(str(meds2List[i]).lower())-j])))
                    kk=1
                    #print("HAHA")
                    break
            for j in range(1,5):
                if (j+(wordTok.index(str(meds2List[i]).lower()))>(numOfWords-2)):
                    #print("HAHA")
                    break
                if (('mg' in str(wordTok[wordTok.index(str(meds2List[i]).lower())+j])) and (kk==0)):
                    mgInfo.insert(i,str(str(wordTok[wordTok.index(str(meds2List[i]).lower())+j-1]))+str(wordTok[wordTok.index(str(meds2List[i]).lower())+j]))
                    #print("HAHA")
                    break
                    
# =============================================================================
#             for k in range(1,10):
#                 if (k+(wordTok.index(str(meds2List[i]).lower())+k)>(numOfWords-1)):
#                     break
#                 if 'count' in str(wordTok[wordTok.index(str(meds2List[i]).lower())+k]):
#                     if(wordTok[wordTok.index(str(meds2List[i]).lower())+k+1]==',') or (wordTok[wordTok.index(str(meds2List[i]).lower())+k+1]=='.'):
#                         taggedList2=stanfordNER2.posTagger(str(wordTok[wordTok.index(str(meds2List[i]).lower())+k+2]))
#                         marker=2
#                     else:
#                         taggedList2=stanfordNER2.posTagger(str(wordTok[wordTok.index(str(meds2List[i]).lower())+k+1]))
#                         marker=1
#                     #print(taggedList2)
#                     if (taggedList2[0][1]=='CD'):
#                         if(marker==1):
#                             countInfo.insert(i,wordTok[wordTok.index(str(meds2List[i]).lower())+k+1])
#                         elif(marker==2):
#                             countInfo.insert(i,wordTok[wordTok.index(str(meds2List[i]).lower())+k+2])
#                     else:
#                         countInfo.insert(i,wordTok[wordTok.index(str(meds2List[i]).lower())+k-1])
#                     break
# =============================================================================

    print(meds2List)    
    print(mgInfo)
    #print(countInfo)
    
    #medsGivenFlag=1
    
    #for sents in sentenceList:
         #if (("Medication is" in sents) or ("medication is" in sents)):
             
             #sntnceeLst.append(sents.lower())
             
    #sntnc=sntnc.lower()
    #print(sntnceeLst)
    if(medsGivenFlag==1):
        numOfMeds=len(meds2List)
        attributesMeds=6
        
        medsAdministrated=[[' .' for x in range(attributesMeds)] for y in range(numOfMeds)] #2D LIST, ROW=num of sentences with keyword, COL=attributes which is 8 here.
        #print(proceduresTaken)
        
        for i in range(numOfMeds):
            print(meds2List[i])
            medsAdministrated[i].insert(0,meds2List[i])
            if (countInfo[i]==' '):
                medsAdministrated[i].insert(1,str(mgInfo[i])+" "+str(countInfo[i]))
            else:
                medsAdministrated[i].insert(1,str(mgInfo[i])+" X "+str(countInfo[i])+"counts")
# =============================================================================
#             sentence=str(sntnceeLst[i])
#             #print(sentence)
#             wordTok=word_tokenize(sentence)
#             #print(wordTok)
#             
#             if (('medication' in wordTok) and ('is' in wordTok) and ( wordTok.index('is') == wordTok.index('medication')+1 ) ):
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('is')+ii+1)])
#                     if(str(wordTok[(wordTok.index('is')+ii+2)])==','):
#                         break 
#                 medsAdministrated[i].insert(0,tempSntnce)
#             if ('dose' in wordTok):
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('dose')+ii+1)])
#                     if(str(wordTok[(wordTok.index('dose')+ii+2)])==','):
#                         break           
#                 medsAdministrated[i].insert(1,tempSntnce)
#             if ('time' in wordTok):
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('time')+ii+1)])
#                     if(str(wordTok[(wordTok.index('time')+ii+2)])==','):
#                         break 
#                 medsAdministrated[i].insert(2,tempSntnce)
#             if (('employee' in wordTok) and ('id' in wordTok) and ( wordTok.index('id') == wordTok.index('employee')+1 )):
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('id')+ii+1)])
#                     if(str(wordTok[(wordTok.index('id')+ii+2)])==','):
#                         break 
#                 medsAdministrated[i].insert(3,tempSntnce)
#             if ('wasted' in wordTok):
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('wasted')+ii+1)])
#                     if(str(wordTok[(wordTok.index('wasted')+ii+2)])==','):
#                         break 
#                 medsAdministrated[i].insert(4,tempSntnce)
#             if ('witness' in wordTok):
#                 tempSntnce=''
#                 for ii in range(100):
#                     tempSntnce=tempSntnce+' '+str(wordTok[(wordTok.index('witness')+ii+1)])
#                     if(str(wordTok[(wordTok.index('witness')+ii+2)])=='.'):
#                         break 
#                 medsAdministrated[i].insert(5,tempSntnce)
# =============================================================================
        
            
# =============================================================================
#     print(medsAdministrated)
#     for i in range(numOfMeds):
#         for j in range(attributesMeds):
#             print(medsAdministrated[i][j])
#         print("###########")
#         
# =============================================================================

###################################################### signatures
     
    signatures=''
    aicSig=''
    aic_counter=0
    mdSig=''
    md_counter=0
    signatureFlag=0
    for sents in sentenceList:
         if (("Signatures" in sents) or ("signatures" in sents)):
             signatureFlag=1    
             signatures=sents
    
    if(signatureFlag==1):
        taggedList=stanfordNER2.getEntity(signatures,3)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='PERSON' and aic_counter==0):
                aicSig=str(taggedList[i][0])
                aic_counter=1
                continue
            if (taggedList[i][1]=='PERSON' and md_counter==0 and aic_counter==1):
                mdSig=str(taggedList[i][0])
                aic_counter=1
    #print(aicSig,mdSig)
    
##################################################### narcotics
    
    narcotics=''
    narcoticsFlag=0
    for sents in sentenceList:
         if (("Narcotics accounted for" in sents) or ("narcotics accounted for" in sents)):
             narcoticsFlag=1    
             narcotics=sents
    if(narcoticsFlag==1):
        narcotics=narcotics.replace("Narcotics accounted for is","")
        narcotics=narcotics.replace("Narcotics accounted for are","")
        narcotics=narcotics.replace("narcotics accounted for is","")
        narcotics=narcotics.replace("narcotics accounted for are","")
    
    #print(narcotics)
    
##################################################### last line
    
    startingMileage=''
    endingMileage=''
    totalMileage=''
    drugBoxUsed=''
    new=''
    
    flagmileage=0
    flagmileage1=0
    flagmileage2=0
    flagmileage4=0
    
    sntnc=''
    for sents in sentenceList:
        if (("Starting mileage" in sents) or ("starting mileage" in sents)):
            sntnc = sntnc +' '+ sents
            flagmileage=1
    #print(sntnc)
    
    if(flagmileage==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                startingMileage=(taggedList[i][0]) 
    #print(startingMileage)
            
    sntnc=''
    for sents in sentenceList:
        if (("Ending mileage" in sents) or ("ending mileage" in sents)):
            flagmileage1=1
            sntnc = sntnc +' '+ sents
    #print(sntnc)
    if(flagmileage1==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                endingMileage=(taggedList[i][0])
    #print(endingMileage)
    if(flagmileage==1 and flagmileage1==1):
        totalMileage=float(endingMileage)-float(startingMileage)
    else:
        totalMileage=' Unknown'
    #print(totalMileage)
    
    sntnc=''
    drugBoxUsed=' Unknown'
    for sents in sentenceList:
        if (("Drug box used" in sents) or ("drug box used" in sents)):
            sntnc = sntnc +' '+ sents
            flagmileage2=1
    #print(sntnc)
    
    if(flagmileage2==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                drugBoxUsed=(taggedList[i][0])
    #print(drugBoxUsed)

    sntnc=''
    new=' Unknown'
    for sents in sentenceList:
        if (("New count" in sents) or ("new count" in sents)):
            sntnc = sntnc +' '+ sents
            flagmileage4=1
    #print(sntnc)
    if(flagmileage4==1):
        taggedList=stanfordNER2.posTagger(sntnc)
        #print(len(taggedList))
        for i in range(len(taggedList)):
            if (taggedList[i][1]=='CD'):
                new=(taggedList[i][0])
    #print(new)
    
    #print(startingMileage,endingMileage,totalMileage,drugBoxUsed,new)
    
############################################################################### pdf generation
    ###########################################################################
        #######################################################################
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'BU', 20)
    pdf.set_fill_color(255,255,0)
    pdf.cell(0, 10, 'FIRE RESCUE', 0, 1, 'C')
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, 'ALBEMERLE COUNTY', 0, 0, 'L')
    pdf.cell(0, 8, 'INITIAL PATIENT CARE REPORT', 0, 1, 'R')
    
    pdf.set_font('Arial', 'B', 7)
    pdf.cell(60, 3, '460 Stagecoach Drive, Suite F', 0, 0, 'L')
    pdf.cell(0, 3, 'PPCR will be available on', 0, 1, 'R')
    
    pdf.cell(60, 3, 'Charlottesville, VA 22902-6489', 0, 0, 'L')
    pdf.cell(0, 3, 'Hospital Bridge within 24 hours', 0, 1, 'R')
    
    pdf.cell(60, 7, 'Phone: (434)296-5833   -  OEMS Agency #00939', 0, 1, 'L')
    
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'PATIENT INFORMATION', 1, 1, 'L', True)
    
    pdf.set_font('Arial', '', 14)
    pdf.multi_cell(0, 9, 'NAME: '+namePatient2, 1, 1, 'L')
    pdf.cell(0, 9, 'ADDRESS: '+address, 1, 1, 'L')
    pdf.cell(90, 9, 'CITY: '+city, 1, 0, 'L')
    pdf.cell(45, 9, 'STATE: '+state, 1, 0, 'L')
    pdf.cell(55, 9, 'ZIP: '+zipcode, 1, 1, 'L')
    pdf.cell(90, 9, 'DOB: '+dob2, 1, 0, 'L')
    pdf.cell(0, 9, 'SSN: '+ssn2, 1, 1, 'L')
    pdf.cell(45, 9, 'AGE: '+age2, 1, 0, 'L')
    pdf.cell(45, 9, 'SEX: '+sex, 1, 0, 'L')
    
    posX=pdf.get_x()
    posY=pdf.get_y()    
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 9, 'FACILITY: UVA   MJH   OTHER-'+facility, 1, 1, 'L')

    posX1=pdf.get_x()
    posY1=pdf.get_y()
    
    if(facilityFlag==1 or facilityFlag==2 or facilityFlag==3):
        if(facilityFlag==1):
           pdf.set_xy(posX,posY)
           pdf.set_text_color(255,0,0) 
           pdf.cell(0, 9, '                  UVA'+facility, 1, 1, 'L')
        if(facilityFlag==2):
           pdf.set_xy(posX,posY)
           pdf.set_text_color(255,0,0) 
           pdf.cell(0, 9, '                            MJH', 1, 1, 'L')
        if(facilityFlag==3):
           pdf.set_xy(posX,posY)
           pdf.set_text_color(255,0,0) 
           pdf.cell(0, 9, '                                       OTHER-'+facility, 1, 1, 'L')
           
        pdf.set_xy(posX1,posY1)
        pdf.set_text_color(0,0,0)
           
           
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 8, '', 0, 1, 'C')
    pdf.cell(0, 10, 'MEDICAL INFORMATION', 1, 1, 'L', True)
    
    pdf.set_font('Arial', '', 14)
    pdf.multi_cell(0, 8, 'CHIEF COMPLAINT: '+chiefComplaint2, 1, 1, 'L')
    pdf.set_font('Arial', '', 12)
    #pdf.multi_cell(0, 18, 'HPI: '+hpi, 1, 1, 'L')
    pdf.multi_cell(0, 18, 'HPI: '+hpi2, 1, 1, 'L')
    
    posX=pdf.get_x()
    posY=pdf.get_y()
#    pdf.multi_cell(0, 18, 'PMH:  ASTHMA   COPD   CHF   CAD   MI   RENAL FAILURE   CVA   DIABETES   HTN   SZ\n'+pmh, 1, 1, 'J')
    pdf.multi_cell(0, 18, 'PMH:  ASTHMA   COPD   CHF   CAD   MI   RENAL FAILURE   CVA   DIABETES   HTN   SZ\n', 1, 1, 'J')
    
    posX1=pdf.get_x()
    posY1=pdf.get_y()
    #todo work here
    if (asthma==1 or copd==1 or chf==0 or cad==1 or mi==1 or renal_failure==1 or cva==1 or diabetes==1 or htn==1 or sz==1):
        if(asthma==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0, 18, '           ASTHMA\n', 1, 1, 'J')
        if(copd==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0, 18, '                             COPD\n', 1, 1, 'J')
        if(chf==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)  
            pdf.multi_cell(0, 18, '                                          CHF\n', 1, 1, 'J')
        if(mi==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)            
            pdf.multi_cell(0, 18, '                                                               MI\n', 1, 1, 'J')
        if(renal_failure==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)            
            pdf.multi_cell(0, 18, '                                                                      RENAL FAILURE\n', 1, 1, 'J')
        if(cva==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0) 
            pdf.multi_cell(0, 18, '                                                                                                     CVA\n', 1, 1, 'J')
        if(diabetes==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)            
            pdf.multi_cell(0, 18, '                                                                                                                DIABETES\n', 1, 1, 'J')
        if(htn==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)        
            pdf.multi_cell(0, 18, '                                                                                                                                     HTN\n', 1, 1, 'J')
        if(sz==1):
            pdf.set_xy(posX,posY)
            pdf.set_text_color(255,0,0)
            pdf.multi_cell(0, 18, '                                                                                                                                               SZ\n', 1, 1, 'J')
            
        pdf.set_xy(posX1,posY1)
        pdf.set_text_color(0,0,0)
    
    
    pdf.set_text_color(0,0,0)
    pdf.multi_cell(0, 8, 'PMH: '+pmhN, 1, 1, 'L')
    pdf.multi_cell(0, 8, 'MEDS: '+meds22, 1, 1, 'L')
    pdf.multi_cell(0, 8, 'ALLERGIES: '+allergies, 1, 1, 'J')
    pdf.multi_cell(0, 8, 'PE/RX/TX: '+perxtx2, 1, 1, 'J')
    
    pdf.add_page()
    #pdf.set_font('Arial', 'B', 14)
    pdf.set_fill_color(255,255,0)
    pdf.set_font('Arial', '', 16)
    pdf.cell(120, 7, 'CALL INFORMATION', 1, 0, 'L', True)
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 7, 'INCIDENT#: '+incidentID2, 1, 1, 'L')
    
    pdf.cell(70, 7, 'UNIT#: '+unitID2, 1, 0, 'L')
    pdf.cell(50, 7, 'EMP. ID ', 1, 0, 'C')
    pdf.cell(0, 7, 'DATE: '+date2, 1, 1, 'L')
    
    pdf.cell(70, 7, 'AIC: '+aic2, 1, 0, 'L')
    pdf.cell(50, 7, aicEmpID, 1, 0, 'C')
    pdf.cell(0, 7, 'DISPATCHED: '+dispatchedTime, 1, 1, 'L')
    
    pdf.cell(70, 7, 'DRIVER: '+driver2, 1, 0, 'L')
    pdf.cell(50, 7, driverEmpID, 1, 0, 'C')
    pdf.cell(0, 7, 'RESPONDING: '+respondingTime, 1, 1, 'L')
    
    pdf.cell(70, 7, 'ATT1: '+attendant2[0], 1, 0, 'L')  #todo work here
    pdf.cell(50, 7, attendantEmpID[0], 1, 0, 'C')
    pdf.cell(0, 7, 'ON SCENE: '+onSceneTime, 1, 1, 'L')
    
    pdf.cell(70, 7, 'ATT2: '+attendant2[1], 1, 0, 'L')  #todo work here
    pdf.cell(50, 7, attendantEmpID[1], 1, 0, 'C')
    pdf.cell(0, 7, 'PT. CONTACT: '+ptContactTime, 1, 1, 'L')
    
    pdf.set_font('Arial', '', 16)
    pdf.cell(120, 7, 'RESPONSE LOCATION', 1, 0, 'L', True)
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 7, 'LEAVE SCENE: '+leaveSceneTime, 1, 1, 'L')
    
    pdf.cell(120, 7, responseLoc+'    ZIP: '+responseZip, 1, 0, 'C')
    pdf.cell(0, 7, 'ARRIVE DEST.: '+arriveDestTime, 1, 1, 'L')
    
    pdf.cell(70, 7, 'INITIAL LOC: '+initialLoc, 'LTR', 0, 'L', True)
    pdf.cell(50, 7, 'PT. WEIGHT: '+weight2+metric, 'LTR', 0, 'L', True)
    pdf.cell(0, 7, 'LEAVE DEST.:'+leaveDestTime, 1, 1, 'L')
    
    pdf.cell(70, 7, '', 'LRB', 0, 'L')
    pdf.cell(50, 7, '', 'LRB', 0, 'L')
    pdf.cell(0, 7, 'RETURN SERVICE:'+retServiceTime, 1, 1, 'L')
    
    pdf.cell(0, 4, '', 0, 1, 'C')
    pdf.cell(0, 7, 'INITIAL VITAL SIGNS', 1, 1, 'L',True)
    pdf.cell(15, 7, 'TIME', 1, 0, 'C',True)
    pdf.cell(20, 7, primaryInitialVitalSigns[0], 1, 0, 'L')
    pdf.cell(30, 7, 'GLUCOSE', 1, 0, 'C',True)
    pdf.cell(30, 7, primaryInitialVitalSigns[1]+' mg/dl', 1, 0, 'L')
    pdf.cell(15, 7, 'GCS', 1, 0, 'C',True)
    pdf.cell(10, 7, 'E', 1, 0, 'C',True)
    pdf.cell(10, 7, primaryInitialVitalSigns[2], 1, 0, 'C')
    pdf.cell(10, 7, 'V', 1, 0, 'C',True)
    pdf.cell(10, 7, primaryInitialVitalSigns[3], 1, 0, 'C')
    pdf.cell(10, 7, 'M', 1, 0, 'C',True)
    pdf.cell(10, 7, primaryInitialVitalSigns[4], 1, 0, 'C')
    pdf.cell(10, 7, '=', 1, 0, 'C',True)
    pdf.cell(10, 7, primaryInitialVitalSigns[5], 1, 1, 'C')
    
    pdf.cell(15, 7, 'RESP', 1, 0, 'C',True)
    pdf.cell(20, 7, primaryInitialVitalSigns[6], 1, 0, 'L')
    pdf.cell(30, 7, 'BP', 1, 0, 'C',True)
    pdf.cell(30, 7, primaryInitialVitalSigns[7], 1, 0, 'C')
    pdf.cell(15, 7, 'EKG', 1, 0, 'C',True)
    #pdf.cell(10, 10, 'E', 1, 0, 'C',True)
    pdf.cell(0, 7, primaryInitialVitalSigns[8], 1, 1, 'C')
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(15, 7, 'PULSE', 1, 0, 'C',True)
    pdf.set_font('Arial', '', 14)
    pdf.cell(20, 7, primaryInitialVitalSigns[9], 1, 0, 'L')
    pdf.cell(30, 7, 'SPO2', 1, 0, 'C',True)
    pdf.cell(30, 7, primaryInitialVitalSigns[10], 1, 0, 'C')
    pdf.cell(20, 7, 'ETCO2', 1, 0, 'C',True)
    #pdf.cell(10, 10, 'E', 1, 0, 'C',True)
    pdf.cell(35, 7, primaryInitialVitalSigns[11]+'mm Hg', 1, 0, 'R')
    pdf.cell(20, 7, 'TEMP', 1, 0, 'C',True)
    pdf.cell(0, 7, primaryInitialVitalSigns[12], 1, 1, 'C')
    
    #secondary initial vital signs
    pdf.cell(0, 4, '', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(20, 7, 'TIME', 1, 0, 'C',True)
    pdf.cell(15, 7, 'LOC', 1, 0, 'C',True)
    pdf.cell(25, 7, 'PULSE', 1, 0, 'C',True)
    pdf.cell(25, 7, 'RESP', 1, 0, 'C',True)
    pdf.cell(30, 7, 'BP', 1, 0, 'C',True)
    pdf.cell(35, 7, 'EKG', 1, 0, 'C',True)
    pdf.cell(20, 7, 'SPO2', 1, 0, 'C',True)
    pdf.cell(0, 7, 'ETCO2', 1, 1, 'C',True)
    
    for i in range(numOfSecondaryVitalSigns):
        pdf.cell(20, 7, secondaryInitialVitalSigns[i][0], 1, 0, 'C')
        pdf.cell(15, 7, secondaryInitialVitalSigns[i][1], 1, 0, 'C')
        pdf.cell(25, 7, secondaryInitialVitalSigns[i][2], 1, 0, 'C')
        pdf.cell(25, 7, secondaryInitialVitalSigns[i][3], 1, 0, 'C')
        pdf.cell(30, 7, secondaryInitialVitalSigns[i][4], 1, 0, 'C')
        pdf.cell(35, 7, secondaryInitialVitalSigns[i][5], 1, 0, 'C')
        pdf.cell(20, 7, str(secondaryInitialVitalSigns[i][6]+"%"), 1, 0, 'C')
        pdf.cell(0, 7, secondaryInitialVitalSigns[i][7], 1, 1, 'C')

    #procedures
    pdf.cell(0, 4, '', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 7, 'PROCEDURES', 1, 1, 'L',True)
    pdf.cell(25, 7, 'PROCED.', 1, 0, 'C',True)
    pdf.cell(25, 7, 'LOCATION', 1, 0, 'C',True)
    pdf.cell(25, 7, 'SIZE', 1, 0, 'C',True)
    pdf.cell(25, 7, 'ATT.', 1, 0, 'C',True)
    pdf.cell(20, 7, 'SUC.', 1, 0, 'C',True)
    pdf.cell(25, 7, 'TIME', 1, 0, 'C',True)
    pdf.cell(20, 7, 'EMP. ID', 1, 0, 'C',True)
    pdf.cell(0, 7, 'OTHER', 1, 1, 'C',True)
    
    for i in range(numOfProcedures):
        pdf.cell(25, 7, proceduresTaken[i][0], 1, 0, 'C')
        pdf.cell(25, 7, proceduresTaken[i][1], 1, 0, 'C')
        pdf.cell(25, 7, proceduresTaken[i][2], 1, 0, 'C')
        pdf.cell(25, 7, proceduresTaken[i][3], 1, 0, 'C')
        pdf.cell(20, 7, proceduresTaken[i][4], 1, 0, 'C')
        pdf.cell(25, 7, proceduresTaken[i][5], 1, 0, 'C')
        pdf.cell(20, 7, proceduresTaken[i][6], 1, 0, 'C')
        pdf.cell(0, 7, proceduresTaken[i][7], 1, 1, 'C')

    
    #medication administrated
    pdf.cell(0, 4, '', 0, 1, 'C')
    pdf.set_font('Arial', '', 14)
    pdf.cell(0, 7, 'MEDICATIONS ADMINISTRATED:', 1, 1, 'L',True)
    pdf.set_font('Arial', '', 8)
    pdf.cell(45, 7, 'MEDICATION', 1, 0, 'C',True)
    pdf.cell(35, 7, 'DOSE GIVEN/ROUTE', 1, 0, 'C',True)
    pdf.cell(15, 7, 'TIME', 1, 0, 'C',True)
    pdf.cell(15, 7, 'EMP. ID', 1, 0, 'C',True)
    pdf.cell(30, 7, 'AMOUNT WASTED', 1, 0, 'C',True)
    pdf.cell(0, 7, 'WITNESS INT.', 1, 1, 'C',True)
    
    for i in range(numOfMeds):
        pdf.cell(45, 7, medsAdministrated[i][0], 1, 0, 'C')
        pdf.cell(35, 7, medsAdministrated[i][1], 1, 0, 'C')
        pdf.cell(15, 7, medsAdministrated[i][2], 1, 0, 'C')
        pdf.cell(15, 7, medsAdministrated[i][3], 1, 0, 'C')
        pdf.cell(30, 7, medsAdministrated[i][4], 1, 0, 'C')
        pdf.cell(0, 7, medsAdministrated[i][5], 1, 1, 'C')

    
    pdf.cell(0, 4, '', 0, 1, 'C')
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 8, 'SIGNATURES:', 1, 1, 'L',True)
    pdf.cell(80, 18, 'AIC: '+aicSig+'                     MD: '+mdSig, 1, 0, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 18, 'NARCOTIS ACCOUNTED FOR: '+narcotics, 1, 1, 'L')
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 5, 'STARTING MILEAGE:'+startingMileage, 0, 0, 'L')
    pdf.cell(60, 5, 'ENDING MILEAGE:'+endingMileage, 0, 0, 'L')
    pdf.cell(0, 5, 'TOTAL MILEAGE:'+str(totalMileage), 0, 1, 'L')
    pdf.cell(60, 5, 'DRUG BOX USED-#.'+drugBoxUsed, 0, 0, 'L')
    pdf.cell(0, 5, 'NEW:'+new, 0, 1, 'L')
    
# =============================================================================
#     pdf.set_text_color(0,0,0)
#     pdf.multi_cell(0, 8, 'PMH: '+pmhN, 1, 1, 'L')
#     pdf.multi_cell(0, 8, 'MEDS: '+meds22, 1, 1, 'L')
#     pdf.multi_cell(0, 8, 'ALLERGIES: '+allergies, 1, 1, 'J')
#     pdf.multi_cell(0, 20, 'PE/RX/TX: '+perxtx2, 1, 1, 'J')
# =============================================================================
    
    pdf.add_page()
    #pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0,0,0)
    pdf.set_fill_color(255,255,0)
    #pdf.set_font('Arial', '', 16)
    #pdf.cell(120, 7, 'ORIGINAL TRANSCRIPTION', 1, 0, 'L', True)
    #pdf.set_font('Arial', '', 14)
    #pdf.cell(0, 70, ORIGINAL TRANSCRIPTION, 1, 1, 'L')
    
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 8, '', 0, 1, 'C')
    pdf.cell(0, 10, 'ORIGINAL TRANSCRIPTION', 1, 1, 'L', True)
    
    pdf.set_font('Arial', '', 14)
    pdf.multi_cell(0, 7, 'Narrative: '+narrative, 1, 1, 'L')
    
    now=datetime.datetime.now()
    pdf.output(str(caseNo)+".pdf", 'F')
    #pdf.output("fire rescue report for"+namePatient2+" at "+str(now)+".pdf", 'F')
    #pdf.output("fire rescue report for patient.pdf", 'F')
    
        
    
    
    
