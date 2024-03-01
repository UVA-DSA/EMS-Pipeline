import py_trees
import pandas as pd
import numpy as np
from py_trees import blackboard
from scipy import spatial
from py_trees.blackboard import Blackboard
import ConceptExtract as CE
from ranking_func import rank
from collections import defaultdict
import openpyxl


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# dummy leaves


class dummy(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(dummy, self).__init__(name)


class PROTOCOLi_Check(py_trees.behaviour.Behaviour):
    def __init__(self, name='PROTOCOLi Check'):
        super(PROTOCOLi_Check, self).__init__(name)


class PROTOCOLi_Action(py_trees.behaviour.Behaviour):
    def __init__(self, name='PROTOCOLi Action'):
        super(PROTOCOLi_Action, self).__init__(name)


blackboard = Blackboard()

# behaviors in framework


class InformationGathering(py_trees.behaviour.Behaviour):
    def __init__(self, name='Information Extraction',
                 slist="concept_list(s&s)_revised.csv",
                 vlist="Vital_List.csv",
                 exlist="CLfromVt.csv",
                 intlist="concept_list(interventions).csv"):
        super(InformationGathering, self).__init__(name)
        self.slist = slist
        self.vlist = vlist
        self.exlist = exlist
        self.intlist = intlist

    def setup(self, unused_timeout=15):
        '''
        create a ConceptExtractor and initialize the patient status
        the list here is the complete list
        '''
        vcl = pd.read_csv(self.exlist)
        #blackboard = Blackboard()
        global blackboard
        self.sce = CE.ConceptExtractor(self.slist)
        self.sce.StatusInit()
        for item in vcl:
            self.sce.SpecificInit(item)
        self.vce = CE.ConceptExtractor(self.vlist)
        self.ice = CE.ConceptExtractor(self.intlist)
        self.ice.StatusInit()
        self.vce.StatusInit()
        blackboard.Signs = self.sce.Status
        blackboard.Vitals = self.vce.Status
        blackboard.Inters = self.ice.Status
        blackboard.ConcLog = []
        return True

    def Vital2Symptom(self):
        '''
        pulse-85  tachy > 100 extreme tachy >150 bradicardia <= 50
        resp-16  8 - 24 fast slow normal
        bp-164/101  hyper age + 100 hypo < 70 normal mbp =  (sbp + 2*dbp)/3
        gcs-15 decreased mental status <= 14
        glucose-101  hyper >300 hypo <60
        spo2-100 94 - 99
        pain-9 pain severity
        '''
        # pulse to bradycardia or tachycardia
        if len(self.vce.Status['pulse'].content) > 0:
            if self.vce.Status['pulse'].value and self.vce.Status['pulse'].value.isdigit():
                pr = int(self.vce.Status['pulse'].value)

                if pr > 100:
                    self.sce.Status['tachycardia'] = CE.PatientStatus(
                        'tachycardia', True, str(pr), 'pulse')
                    self.sce.Status['tachycardia'].score = self.vce.Status['pulse'].score
                elif pr <= 50:
                    self.sce.Status['bradycardia'] = CE.PatientStatus(
                        'bradycardia', True, str(pr), 'pulse')
                    self.sce.Status['bradycardia'].score = self.vce.Status['pulse'].score
        # resp rate symptoms
        if len(self.vce.Status['resp'].content) > 0:
            if self.vce.Status['resp'].value and self.vce.Status['resp'].value.isdigit():
                rr = int(self.vce.Status['resp'].value)
                if rr > 24:
                    self.sce.Status['tachypnea'] = CE.PatientStatus(
                        'tachypnea', True, str(rr), 'resp')
                    self.sce.Status['tachypnea'].score = self.vce.Status['resp'].score
                elif rr < 8:
                    self.sce.Status['bradypnea'] = CE.PatientStatus(
                        'bradypnea', True, str(rr), 'resp')
                    self.sce.Status['bradypnea'].score = self.vce.Status['resp'].score

        # blood pressure symptoms
        if len(self.vce.Status['bp'].content) > 0:
            if self.vce.Status['bp'].value and '/' in self.vce.Status['bp'].value:
                temp = self.vce.Status['bp'].value.strip().split('/')
                if temp[0] and temp[1] and temp[0].isdigit() and temp[1].isdigit():
                    bp = [int(temp[0]), int(temp[1])]
                    mbp = (bp[0] + 2 * bp[1])/3
                    if bp[0] >= 140 or bp[1] >= 90:
                        self.sce.Status['hypertension'] = CE.PatientStatus(
                            'hypertension', True, str(bp[0]), 'bp')
                        self.sce.Status['hypertension'].score = self.vce.Status['bp'].score
                    elif mbp < 70:
                        self.sce.Status['hypotension'] = CE.PatientStatus(
                            'hypotension', True, str(mbp), 'bp')
                        self.sce.Status['hypotension'].score = self.vce.Status['bp'].score

        # gcs symptoms
        if len(self.vce.Status['gcs'].content) > 0:
            if self.vce.Status['gcs'].value and self.vce.Status['gcs'].value.isdigit():
                gcs = int(self.vce.Status['gcs'].value)
                if gcs < 15:
                    self.sce.Status['decreased mental status'] = CE.PatientStatus(
                        'decreased mental status', True, str(gcs), 'gcs')
                    self.sce.Status['decreased mental status'].score = self.vce.Status['gcs'].score

        # glucose symptoms
        if len(self.vce.Status['glucose'].content) > 0:
            if self.vce.Status['glucose'].value and self.vce.Status['glucose'].value.isdigit():
                glu = int(self.vce.Status['glucose'].value)
                if glu > 300:
                    self.sce.Status['hyperglycemia'] = CE.PatientStatus(
                        'hyperglycemia', True, str(glu), 'glucose')
                    self.sce.Status['hyperglycemia'].score = self.vce.Status['glucose'].score
                elif glu < 60:
                    self.sce.Status['hypoglycemia'] = CE.PatientStatus(
                        'hypoglycemia', True, str(glu), 'glucose')
                    self.sce.Status['hypoglycemia'].score = self.vce.Status['glucose'].score

        # spo2 symptoms
        if len(self.vce.Status['spo2'].content) > 0:
            if self.vce.Status['spo2'].value and self.vce.Status['spo2'].value.isdigit():
                spo2 = int(self.vce.Status['spo2'].value)
                if spo2 < 94:
                    self.sce.Status['hypoxemia'] = CE.PatientStatus(
                        'hypoxemia', True, str(spo2), 'spo2')
                    self.sce.Status['hypoxemia'].score = self.vce.Status['spo2'].score

        # pain
        if len(self.vce.Status['pain'].content) > 0:
            if self.vce.Status['pain'].value and self.vce.Status['pain'].value.isdigit():
                ps = int(self.vce.Status['pain'].value)
                self.sce.Status['pain severity'] = CE.PatientStatus(
                    'pain severity', True, str(ps), 'pain')
                self.sce.Status['pain severity'].score = self.vce.Status['pain'].score

        # ekg symptoms
        if len(self.vce.Status['ekg'].content) > 0:
            ekg = self.vce.Status['ekg'].value
            if "Sinus_Arrhythmia" in ekg:
                self.sce.Status['dysrhythmia'] = CE.PatientStatus('dysrhythmia', True, ekg, 'ekg')
                self.sce.Status['dysrhythmia'].score = self.vce.Status['ekg'].score
            else:
                self.sce.Status['dysrhythmia'] = CE.PatientStatus('dysrhythmia', False, ekg, 'ekg')
                self.sce.Status['dysrhythmia'].score = self.vce.Status['ekg'].score
            if "_Bradycardia" in ekg:
                self.sce.Status['bradycardia'] = CE.PatientStatus('bradycardia', True, ekg, 'ekg')
                self.sce.Status['bradycardia'].score = self.vce.Status['ekg'].score
            if "_Tachycardia" in ekg:
                self.sce.Status['tachycardia'] = CE.PatientStatus('tachycardia', True, ekg, 'ekg')
                self.sce.Status['tachycardia'].score = self.vce.Status['ekg'].score

        return True

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        # print("global tick_num: ", blackboard.tick_num)
        self.sce.ConceptExtract(blackboard.text)

        blackboard.concepts = self.sce.concepts
        blackboard.confi = self.sce.scores
        self.sce.FirstExtract(blackboard.text, blackboard.tick_num)

        self.vce.concepts = blackboard.concepts
        self.vce.scores = self.sce.scores
        self.vce.FirstExtract(blackboard.text, blackboard.tick_num)
        blackboard.Vitals = self.vce.Status
        self.Vital2Symptom()
        blackboard.Signs = self.sce.Status
        # self.sce.DisplayStatus()
        self.ice.concepts = blackboard.concepts
        self.ice.scores = self.sce.scores
        self.ice.FirstExtract(blackboard.text, blackboard.tick_num)
        blackboard.Inters = self.ice.Status
       # self.ice.DisplayStatus()
        return py_trees.common.Status.SUCCESS


class Vectorize(py_trees.behaviour.Behaviour):
    def __init__(self, name='Vectorize', protocols='ODEMSA_Protocols_weighted.xlsx'):
        super(Vectorize, self).__init__(name)
        self.protocols = protocols

    def setup(self, unused_timeout=15):
        #blackboard = Blackboard()
        global blackboard
        PC = dict()
        pro_df = pd.read_excel(self.protocols, engine="openpyxl")
        for line in pro_df.iterrows():
            line_ss = [(i.strip().lower()[:-1], i.strip().lower()[-1])
                       for i in line[1]['Signs&Symptoms'].split(';')]
            if not pd.isnull(line[1]['Possible signs&symptoms additions']):
                line_ssr = [(i.strip().lower()[:-1], i.strip().lower()[-1])
                            for i in line[1]['Possible signs&symptoms additions'].split(';')]
            name = line[1]['Protocol']
            PC[name] = line_ss + line_ssr
        self.PV = dict()
        for item in PC:  # PC is a dictionary where the keys are the protocols and the values are a list of the signs and symptoms
            vec = list()
            su = 0.
            for i in blackboard.Signs:
                res = 0.
                for j in PC[item]:
                    if i == j[0]:
                        res = 8.**int(j[1])
                        break
                su += res
                vec.append(res)
            for i in range(len(vec)):
                vec[i] = vec[i] / su
            self.PV[item] = vec
        blackboard.PV = self.PV
        return True

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        # mm confidence encoding
        TV = []
        for item in blackboard.Signs:
            if blackboard.Signs[item].binary:
                TV.append(blackboard.Signs[item].score / 1000.)
            else:
                TV.append(0.)

        blackboard.TV = TV
        maxsim = 0
        result = ''
        blackboard.ranking = []
        for key in self.PV:
            sim = 1 - spatial.distance.cosine(TV, self.PV[key])
            blackboard.ranking.append((key, sim))
            if sim > maxsim:
                maxsim = sim
                result = key
        blackboard.protocol = result
        blackboard.candi, blackboard.pos = rank(blackboard.ranking)[0], rank(blackboard.ranking)[1]
        return py_trees.common.Status.SUCCESS


class ProtocolSelector(py_trees.behaviour.Behaviour):
    def __init__(self, name='Protocol Selector'):
        super(ProtocolSelector, self).__init__(name)

    def setup(self, unused_timeout=15):
        # blackboard = Blackboard()
        global blackboard
        blackboard.protocol_flag = dict()
        blackboard.feedback = dict()
        for i in blackboard.PV:
            blackboard.protocol_flag[i] = (False, 0.)
        for i in blackboard.Inters:
            blackboard.feedback[i] = 0.
        return True

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        blackboard.protocol_flag = dict()
        blackboard.feedback = dict()
        for i in blackboard.PV:
            blackboard.protocol_flag[i] = (False, 0.)
        for i in blackboard.Inters:
            blackboard.feedback[i] = 0.
        num = sum(blackboard.pos[:3])
        for idx, item in enumerate(blackboard.candi):
            if idx < 3:
                blackboard.protocol_flag[item] = (True, blackboard.pos[idx]/num)
        return py_trees.common.Status.SUCCESS


class TextCollection(py_trees.behaviour.Behaviour):
    def __init__(self, name='Speech To Text Conversion'):
        super(TextCollection, self).__init__(name)

    def setup(self, unused_timeout=15):
        level = 'I/P'
        #blackboard = Blackboard()
        global blackboard
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
        return py_trees.common.Status.SUCCESS

# protocols' checker


class Chestpain_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='Chestpain_Checker'):
        super(Chestpain_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][0]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class Abdopain_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='Abdopain_Checker'):
        super(Abdopain_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][1]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class Behavioral_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='Behavioral_Checker'):
        super(Behavioral_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][2]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class Seizure_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='Seizure_Checker'):
        super(Seizure_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][3]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class resp_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='resp_Checker'):
        super(resp_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][4]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class AMS_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='AMS_Checker'):
        super(AMS_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][5]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class Diab_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='Diab_Checker'):
        super(Diab_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][6]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class Overdose_Checker(py_trees.behaviour.Behaviour):
    def __init__(self, name='Overdose_Checker'):
        super(Overdose_Checker, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][7]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        if blackboard.protocol_flag[self.key][0]:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

# protocols


class Chestpain(py_trees.behaviour.Behaviour):
    def __init__(self, name='Chestpain'):
        super(Chestpain, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][0]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.common.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        if blackboard.candi[0] == self.key and blackboard.Vitals['gcs'].value == '15':
            blackboard.feedback['aspirin'] += self.posi
        # consider add dependency on iv
        if blackboard.Signs['substance abuse history'].binary:
            blackboard.feedback['midazolam'] += self.posi * \
                blackboard.Signs['substance abuse history'].score / 1000.
           # blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['substance abuse history'].score / 1000.
        elif blackboard.Signs['abuse of substance'].binary:
            blackboard.feedback['midazolam'] += self.posi * \
                blackboard.Signs['abuse of substance'].score / 1000.
          #  blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['abuse of substance'].score / 1000.
        if 'STEMI' in blackboard.Vitals['ekg'].value and blackboard.Vitals['pain'].binary \
                and blackboard.Vitals['bp'].binary and int(blackboard.Vitals['bp'].value.strip().split('/')[0]) > 100:
            blackboard.feedback['nitroglycerin'] += self.posi * \
                blackboard.Vitals['pain'].score / 1000. * blackboard.Vitals['bp'].score / 1000.
        if blackboard.Vitals['pain'].binary and blackboard.Inters['nitroglycerin'].binary:
          #  blackboard.feedback['morphine'] += self.posi * blackboard.Vitals['pain'].score / 1000. * blackboard.Inters['nitroglycerin'].score / 1000.
            blackboard.feedback['fentanyl'] += self.posi * blackboard.Vitals['pain'].score / \
                1000. * blackboard.Inters['nitroglycerin'].score / 1000.
        return py_trees.common.Status.SUCCESS


CP_C = Chestpain_Checker()
CP_A = Chestpain()
CP = py_trees.composites.Sequence("Chestpain", children=[CP_C, CP_A])


class Abdopain(py_trees.behaviour.Behaviour):
    def __init__(self, name='Abdopain'):
        super(Abdopain, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][1]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.common.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        blackboard.feedback['normal saline'] += self.posi
        if blackboard.Signs['nausea'].binary:
            blackboard.feedback['ondansetron'] += self.posi * \
                blackboard.Signs['nausea'].score / 1000.
        elif blackboard.Signs['vomiting'].binary:
            blackboard.feedback['ondansetron'] += self.posi * \
                blackboard.Signs['vomiting'].score / 1000.
        return py_trees.common.Status.SUCCESS


AP_C = Abdopain_Checker()
AP_A = Abdopain()
AP = py_trees.composites.Sequence("Abdopain", children=[AP_C, AP_A])


class Behavioral(py_trees.behaviour.Behaviour):
    def __init__(self, name='Behavioral_Checker'):
        super(Behavioral, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][2]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.common.Status.SUCCESS
        if blackboard.Signs['combative'].binary:
            s = blackboard.Signs['combative'].score / 1000.
            if blackboard.Signs['hypoxemia'].binary:
                blackboard.feedback['oxygen'] += self.posi * \
                    blackboard.Signs['hypoxemia'].score / 1000. * s
            blackboard.feedback['physical restraint'] += self.posi * s
            if blackboard.Signs['agitation'].binary:
                blackboard.feedback['midazolam'] += self.posi * \
                    blackboard.Signs['agitation'].score / 1000. * s
             #   blackboard.feedback['diazepam'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
              #  blackboard.feedback['geodon'] += self.posi * blackboard.Signs['agitation'].score / 1000. * s
            blackboard.feedback['transport'] += self.posi * s
        elif blackboard.Signs['violent'].binary:
            s = blackboard.Signs['violent'].score / 1000.
            if blackboard.Signs['hypoxemia'].binary:
                blackboard.feedback['oxygen'] += self.posi * \
                    blackboard.Signs['hypoxemia'].score / 1000. * s
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
        return py_trees.common.Status.SUCCESS


BE_C = Behavioral_Checker()
BE_A = Behavioral()
BE = py_trees.composites.Sequence("Behavioral", children=[BE_C, BE_A])


class Seizure(py_trees.behaviour.Behaviour):
    def __init__(self, name='Seizure'):
        super(Seizure, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][3]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.behaviours.Success
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['normal saline'] += self.posi
        # print("skitipppapa")
        # print(blackboard.Vitals['glucose'].value)
        # print("hello")
        if not blackboard.Signs['hypotension'].binary and blackboard.Signs['seizure'].binary and blackboard.Vitals['glucose'].binary and blackboard.Vitals['glucose'].value > 60:
            s = blackboard.Signs['seizure'].score / 1000. * blackboard.Vitals['glucose'].score / \
                1000. * blackboard.Signs['hypotension'].score / 1000.
            blackboard.feedback['midazolam'] += self.posi * s
          #  blackboard.feedback['diazepam'] += self.posi * s
        blackboard.feedback['cardiac monitor'] += self.posi
        # recovery position?
        blackboard.feedback['transport'] += self.posi
        return py_trees.common.Status.SUCCESS


SZ_C = Seizure_Checker()
SZ_A = Seizure()
SZ = py_trees.composites.Sequence("Seizure", children=[SZ_C, SZ_A])


class resp(py_trees.behaviour.Behaviour):
    '''
    Acute respiratory distress syndrome:
    sob: shortness of breath
    rapid breathing: tachypnea
    low bp: hypotension
    confusion: confusion
    '''

    def __init__(self, name='resp'):
        super(resp, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][4]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.behaviours.Success
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        # have to do?
       # blackboard.feedback['capnography'] += self.posi
       # blackboard.feedback['bronchodilator'] += self.posi
    #    blackboard.feedback['metered dose inhaler'] += self.posi
        # if respiratory distress
        if blackboard.Signs['shortness of breath'].binary and (blackboard.Signs['hypoxemia'].binary or blackboard.Signs['rhonchi'].binary):
            s = blackboard.Signs['shortness of breath'].score / 1000. *\
                ((blackboard.Signs['hypoxemia'].binary * blackboard.Signs['hypoxemia'].score +
                  blackboard.Signs['rhonchi'].binary * blackboard.Signs['rhonchi'].score) /
                 (blackboard.Signs['hypoxemia'].binary + blackboard.Signs['rhonchi'].binary) / 1000.)
            blackboard.feedback['bag valve mask ventilation'] += self.posi * s
            # bvm ventilation have been done
            if blackboard.Inters['bag valve mask ventilation'].binary:
                blackboard.feedback['endotracheal tube'] += self.posi * s * \
                    blackboard.Inters['bag valve mask ventilation'].score / 1000.
            if not blackboard.Signs['hypertension'].binary:
                blackboard.feedback['albuterol'] += self.posi * s * \
                    blackboard.Signs['hypertension'].score / 1000.
                blackboard.feedback['ipratropium'] += self.posi * \
                    s * blackboard.Signs['hypertension'].score / 1000.
            if blackboard.Signs['wheezing'].binary:
                blackboard.feedback['dexamethasone'] += self.posi * \
                    s * blackboard.Signs['wheezing'].score / 1000.
            if blackboard.Signs['hypoxemia'].binary and blackboard.Signs['tachypnea'].binary \
                    and blackboard.Vitals['bp'].binary and blackboard.Vitals['bp'].value.strip().split('/')[0].isdigit() and int(blackboard.Vitals['bp'].value.strip().split('/')[0]) > 90:
                blackboard.feedback['cpap'] += self.posi * s * \
                    blackboard.Signs['tachypnea'].score / 1000.
            blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi

        return py_trees.common.Status.SUCCESS


RE_C = resp_Checker()
RE_A = resp()
RE = py_trees.composites.Sequence("resp", children=[RE_C, RE_A])


class AMS(py_trees.behaviour.Behaviour):
    def __init__(self, name='AMS'):
        super(AMS, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][5]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.behaviours.Success
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['normal saline'] += self.posi
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        return py_trees.common.Status.SUCCESS


AMS_C = AMS_Checker()
AMS_A = AMS()
AMS = py_trees.composites.Sequence("resp", children=[AMS_C, AMS_A])


class Diab(py_trees.behaviour.Behaviour):
    def __init__(self, name='Diab'):
        super(Diab, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][6]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            return py_trees.common.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        if blackboard.Signs['hypoglycemia'].binary:
            s = blackboard.Signs['hypoglycemia'].score / 1000.
            if blackboard.Signs['loss of consciousness'].binary:
                blackboard.feedback['normal saline'] += self.posi * \
                    blackboard.Signs['loss of consciousness'].score / 1000. * s
                blackboard.feedback['dextrose'] += self.posi * \
                    blackboard.Signs['loss of consciousness'].score / 1000. * s
            else:
                blackboard.feedback['oral glucose'] += self.posi * s
        return py_trees.common.Status.SUCCESS


DI_C = Diab_Checker()
DI_A = Diab()
DI = py_trees.composites.Sequence("Diab", children=[DI_C, DI_A])


class Overdose(py_trees.behaviour.Behaviour):
    def __init__(self, name='Overdose'):
        super(Overdose, self).__init__(name)
        p = pd.read_excel('ODEMSA_Protocols_weighted.xlsx', engine="openpyxl")
        self.key = p['Protocol'][7]

    def update(self):
        #blackboard = Blackboard()
        global blackboard
        self.posi = blackboard.protocol_flag[self.key][1]
        if self.posi == 0:
            py_trees.common.Status.SUCCESS
        if blackboard.Signs['hypoxemia'].binary:
            blackboard.feedback['oxygen'] += self.posi * blackboard.Signs['hypoxemia'].score / 1000.
        if blackboard.Vitals['bp'].binary and "/" in blackboard.Vitals['bp'].value and int(blackboard.Vitals['bp'].value.strip().split('/')[0]) <= 90:
            blackboard.feedback['normal saline'] += self.posi * \
                blackboard.Vitals['bp'].score / 1000.
        if blackboard.Signs['abuse of substance'].binary:
            blackboard.feedback['narcan'] += self.posi * \
                blackboard.Signs['abuse of substance'].score / 1000.
        elif blackboard.Signs['pin point pupils'].binary:
            blackboard.feedback['narcan'] += self.posi * \
                blackboard.Signs['pin point pupils'].score / 1000.
        elif blackboard.Signs['decreased mental status'].binary:
            blackboard.feedback['narcan'] += self.posi * \
                blackboard.Signs['decreased mental status'].score / 1000.
        elif blackboard.Signs['hypotension'].binary:
            blackboard.feedback['narcan'] += self.posi * \
                blackboard.Signs['hypotension'].score / 1000.
        elif blackboard.Signs['bradypnea'].binary:
            blackboard.feedback['narcan'] += self.posi * blackboard.Signs['bradypnea'].score / 1000.
        blackboard.feedback['cardiac monitor'] += self.posi
        blackboard.feedback['transport'] += self.posi
        return py_trees.common.Status.SUCCESS


OV_C = Overdose_Checker()
OV_A = Overdose()
OV = py_trees.composites.Sequence("Diab", children=[OV_C, OV_A])

protocols = py_trees.composites.Parallel('protocols',
                                         children=[CP, OV, DI, AMS, RE, SZ, BE, AP])
