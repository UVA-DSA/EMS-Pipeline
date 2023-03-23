from collections import defaultdict
import random
import os,sys
import torch
import numpy as np
import time
import json
import itertools

date = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
# date = '2023-02-08-22_08_09'

max_iter = 1
terminate = False
lamb = 0.5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
head_p_node, mid_p_node, tail_p_node = None, None, None
p_node = [
    'general - cardiac arrest',
    'injury - general trauma management (protocol 4 - 1)',
    'medical - abdominal pain (protocol 3 - 2)',
    'medical - altered mental status',
    'medical - chest pain - cardiac suspected (protocol 2 - 1)',
    'medical - diabetic - hypoglycemia',
    'medical - overdose/poisoning - opioid (protocol 7 - 1)',
    'medical - overdose/poisoning/toxic ingestion (protocol 9 - 9)',
    'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress) (protocol 3 - 11)',
    'medical - seizure',
    'medical - stroke/tia (protocol 3 - 5)'
]

# p_node = [
#      'airway - failed (protocol 3 - 16)',
#      'airway - obstruction/foreign body (protocol 9 - 4)',
#      'environmental - heat exposure/heat exhaustion environmental - heat stroke (protocol 5 - 2)',
#      'environmental - hypothermia (protocol 5 - 1)',
#      'general - behavioral/patient restraint (protocol 3 - 4)',
#      'general - cardiac arrest',
#      'general - fever (protocol 9 - 3)',
#      'general - pain control',
#      'hemorrhage control (protocol 4 - 14)',
#      'injury - abdominal trauma',
#      'injury - burns - thermal',
#      'injury - general trauma management (protocol 4 - 1)',
#      'injury - head',
#      'injury - thoracic (protocol 4 - 11)',
#      'medical - abdominal pain (protocol 3 - 2)',
#      'medical - allergic reaction/anaphylaxis',
#      'medical - altered mental status',
#      'medical - bradycardia',
#      'medical - chest pain - cardiac suspected (protocol 2 - 1)',
#      'medical - diabetic - hyperglycemia',
#      'medical - diabetic - hypoglycemia',
#      'medical - hypotension/shock (nontrauma)',
#      'medical - nausea/vomiting',
#      'medical - overdose/poisoning - calcium channel blocker (protocol 7 - 5)',
#      'medical - overdose/poisoning - opioid (protocol 7 - 1)',
#      'medical - overdose/poisoning - stimulant (protocol 7 - 2)',
#      'medical - overdose/poisoning/toxic ingestion (protocol 9 - 9)',
#      'medical - pulmonary edema/chf (protocol 2 - 3)',
#      'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress) (protocol 3 - 11)',
#      'medical - seizure',
#      'medical - st elevation myocardial infarction (acs/ami) (protocol 2 - 2)',
#      'medical - stroke/tia (protocol 3 - 5)',
#      'medical - supraventricular tachycardia (including atrial fibrillation) medical - tachycardia medical - ventricular tachycardia with a pulse',
#      'ob/gyn - childbirth/labor/delivery (protocol 6 - 2)',
#      'ob/gyn - pre-term labor (pre-term labor) (protocol 6 - 12)',
#      'spinal motion restriction (protocol 4 - 13)'
# ]


'''
# PROTOCOLS TRAIN NUMBER (0, 10]
tail_p_node = [
    'airway - obstruction/foreign body (protocol 9 - 4)',
    'environmental - hypothermia (protocol 5 - 1)',
    'injury - abdominal trauma',
    'injury - crush syndrome (protocol 4 - 4)',
    'injury - thoracic (protocol 4 - 11)',
    'medical - overdose/poisoning - calcium channel blocker (protocol 7 - 5)',
    'medical - overdose/poisoning - stimulant (protocol 7 - 2)',
    'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress - croup/epiglottits) (protocol 9 - 11)'
]

# PROTOCOLS TRAIN NUMBER (10, 100]
mid_p_node = [
    'airway - failed (protocol 3 - 16)',
    'environmental - heat exposure/heat exhaustion environmental - heat stroke (protocol 5 - 2)',
    'general - behavioral/patient restraint (protocol 3 - 4)',
    'general - fever (protocol 9 - 3)',
    'general - pain control',
    'hemorrhage control (protocol 4 - 14)',
    'injury - burns - thermal',
    'injury - head',
    'medical - allergic reaction/anaphylaxis',
    'medical - bradycardia',
    'medical - diabetic - hyperglycemia',
    'medical - hypotension/shock (nontrauma)',
    'medical - nausea/vomiting',
    'medical - pulmonary edema/chf (protocol 2 - 3)',
    'medical - st elevation myocardial infarction (acs/ami) (protocol 2 - 2)',
    'medical - supraventricular tachycardia (including atrial fibrillation) medical - tachycardia medical - ventricular tachycardia with a pulse',
    'ob/gyn - childbirth/labor/delivery (protocol 6 - 2)',
    'ob/gyn - pre-term labor (pre-term labor) (protocol 6 - 12)',
    'spinal motion restriction (protocol 4 - 13)'
]

# PROTOCOLS TRAIN NUMBER (100, inf)
head_p_node = [
    'general - cardiac arrest',
    'injury - general trauma management (protocol 4 - 1)',
    'medical - abdominal pain (protocol 3 - 2)',
    'medical - altered mental status',
    'medical - chest pain - cardiac suspected (protocol 2 - 1)',
    'medical - diabetic - hypoglycemia',
    'medical - overdose/poisoning - opioid (protocol 7 - 1)',
    'medical - overdose/poisoning/toxic ingestion (protocol 9 - 9)',
    'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress) (protocol 3 - 11)',
    'medical - seizure',
    'medical - stroke/tia (protocol 3 - 5)'
]

p_node = list(itertools.chain(head_p_node, mid_p_node, tail_p_node))
'''

tmp_dict = {
    'Medical - Diabetic - Hyperglycemia (Protocol 3 - 7)': 'Medical - Diabetic - Hyperglycemia',
    'Medical - Diabetic - Hyperglycemia (Protocol 9 - 5)': 'Medical - Diabetic - Hyperglycemia',
    'General - Cardiac Arrest (Protocol 2 - 7)': 'General - Cardiac Arrest',
    'General - Cardiac Arrest (Protocol 8 - 2)': 'General - Cardiac Arrest',
    'Medical - Newborn/Neonatal Resuscitation (Protocol 6 - 3)': 'Medical - Newborn/Neonatal Resuscitation',
    'Medical - Newborn/Neonatal Resuscitation (Protocol 8 - 3)': 'Medical - Newborn/Neonatal Resuscitation',
    'Cardiac Arrest - Unknown Rhythm (i.e. BLS) (Protocol 2 - 6)': 'Cardiac Arrest - Unknown Rhythm (i.e. BLS)',
    'Cardiac Arrest - Unknown Rhythm (i.e. BLS) (Protocol 8 - 1)': 'Cardiac Arrest - Unknown Rhythm (i.e. BLS)',
    'Medical - Hypotension/Shock (Nontrauma) (Protocol 3 - 13)': 'Medical - Hypotension/Shock (Nontrauma)',
    'Medical - Hypotension/Shock (Nontrauma) (Protocol 9 - 13)': 'Medical - Hypotension/Shock (Nontrauma)',
    'Medical - Diabetic - Hypoglycemia (Protocol 3 - 8)': 'Medical - Diabetic - Hypoglycemia',
    'Medical - Diabetic - Hypoglycemia (Protocol 9 - 6)': 'Medical - Diabetic - Hypoglycemia',
    'Medical - Bradycardia (Protocol 2 - 9)': 'Medical - Bradycardia',
    'Medical - Bradycardia (Protocol 8 - 5)': 'Medical - Bradycardia',
    'Medical - Nausea/Vomiting (Protocol 3 - 9)': 'Medical - Nausea/Vomiting',
    'Medical - Nausea/Vomiting (Protocol 9 - 7)': 'Medical - Nausea/Vomiting',
    'Injury - Burns - Thermal (Protocol 4 - 3)': 'Injury - Burns - Thermal',
    'Injury - Burns - Thermal (Protocol 10 - 2)': 'Injury - Burns - Thermal',
    'Medical - Allergic Reaction/Anaphylaxis (Protocol 3 - 3)': 'Medical - Allergic Reaction/Anaphylaxis',
    'Medical - Allergic Reaction/Anaphylaxis (Protocol 9 - 2)': 'Medical - Allergic Reaction/Anaphylaxis',
    'Injury - Head (Protocol 4 - 6)': 'Injury - Head',
    'Injury - Head (Protocol 10 - 4)': 'Injury - Head',
    'Injury - Electrical Injuries (Protocol 4 - 5)': 'Injury - Electrical Injuries',
    'Injury - Electrical Injuries (Protocol 10 - 3)': 'Injury - Electrical Injuries',
    'Medical - Seizure (Protocol 3 - 12)': 'Medical - Seizure',
    'Medical - Seizure (Protocol 9 - 12)': 'Medical - Seizure',
    'Medical - Altered Mental Status (Protocol 3 - 15)': 'Medical - Altered Mental Status',
    'Medical - Altered Mental Status (Protocol 9 - 14)': 'Medical - Altered Mental Status',
    'General - Pain Control (Protocol 3 - 10)': 'General - Pain Control',
    'General - Pain Control (Protocol 9 - 8)': 'General - Pain Control',
    'General - Universal Patient Care/Initial Patient Contact (Medical Patient Assessment) (Protocol 3 - 1)': 'General - Universal Patient Care/Initial Patient Contact',
    'General - Universal Patient Care/Initial Patient Contact (Pediatric Medical Patient Assessment) (Protocol 9 - 1)': 'General - Universal Patient Care/Initial Patient Contact',
    'Injury - Abdominal Trauma (Abdominal Trauma) (Protocol 4 - 2)': 'Injury - Abdominal Trauma',
    'Injury - Abdomen PEDIATRIC TRAUMA EMERGENCIES (Abdominal Trauma) (Protocol 10 - 1)': 'Injury - Abdominal Trauma',
    'Medical - Supraventricular Tachycardia (including atrial fibrillation) Medical - Tachycardia Medical - Ventricular Tachycardia with a Pulse (Protocol 2 - 8)': 'Medical - Supraventricular Tachycardia (including atrial fibrillation) Medical - Tachycardia Medical - Ventricular Tachycardia with a Pulse',
    'Medical - Supraventricular Tachycardia (including atrial fibrillation) Medical - Tachycardia Medical - Ventricular Tachycardia with a Pulse (Protocol 8 - 4)': 'Medical - Supraventricular Tachycardia (including atrial fibrillation) Medical - Tachycardia Medical - Ventricular Tachycardia with a Pulse'
}
group_p_dict = {}
for k, v in tmp_dict.items():
    group_p_dict[k.lower()] = v.lower()

reverse_group_p_dict = defaultdict(list)
for k, v in tmp_dict.items():
    reverse_group_p_dict[v.lower()].append(k.lower())

ungrouped_p_node = []
for p in p_node:
    p = p.lower().strip()
    tmp = []
    if p in reverse_group_p_dict:
        tmp = reverse_group_p_dict[p]
    else:
        tmp.append(p)
    for i in range(len(tmp)):
        if tmp[i] not in ungrouped_p_node:
            ungrouped_p_node.append(tmp[i])

print(os.getcwd())
with open('./EMSAgent/Interface/config_file/json files/Hierarchy/hier2p.json', 'r') as f:
    hier2p = json.load(f)
with open('./EMSAgent/Interface/config_file/json files/Hierarchy/p2hier.json', 'r') as f:
    p2hier = json.load(f)

hier = []
for p in ungrouped_p_node:
    cur_hier = p2hier[p].lower()
    if cur_hier not in hier:
        hier.append(cur_hier)
hier = sorted(hier)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True