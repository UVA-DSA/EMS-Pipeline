from collections import defaultdict
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np
import time
import json
import itertools
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
date = '2023-08-10-11_40_22'

task = 'multi_label'
dataset = 'EMS' #EMS, MIMIC3
multi_graph = False #if use multi graph (KAMG)
groupby = 'age'  #['age', 'hierarchy', None]
device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_RESULT_ROOT = os.path.dirname(os.path.realpath(__file__))
# SAVE_RESULT_ROOT = '/gpfs/gpfs0/scratch/zar8jw'

if dataset == 'EMS':
    EMS_DIR = None
    # EMS_DIR = '/sfs/qumulo/qhome/zar8jw/EMS/data/EMS/multi_label'
    p_node = [
        'airway - failed (protocol 3 - 16)',
        'environmental - heat exposure/heat exhaustion environmental - heat stroke (protocol 5 - 2)',
        'environmental - hypothermia (protocol 5 - 1)',
        'general - behavioral/patient restraint (protocol 3 - 4)',
        'general - cardiac arrest',
        'general - pain control',
        'hemorrhage control (protocol 4 - 14)',
        'injury - abdominal trauma',
        'injury - burns - thermal',
        'injury - crush syndrome (protocol 4 - 4)',
        'injury - general trauma management (protocol 4 - 1)',
        'injury - head',
        'injury - thoracic (protocol 4 - 11)',
        'medical - abdominal pain (protocol 3 - 2)',
        'medical - allergic reaction/anaphylaxis',
        'medical - altered mental status',
        'medical - bradycardia',
        'medical - chest pain - cardiac suspected (protocol 2 - 1)',
        'medical - diabetic - hyperglycemia',
        'medical - diabetic - hypoglycemia',
        'medical - hypotension/shock (nontrauma)',
        'medical - nausea/vomiting',
        'medical - overdose/poisoning - calcium channel blocker (protocol 7 - 5)',
        'medical - overdose/poisoning - opioid (protocol 7 - 1)',
        'medical - overdose/poisoning - stimulant (protocol 7 - 2)',
        'medical - overdose/poisoning/toxic ingestion (protocol 9 - 9)',
        'medical - pulmonary edema/chf (protocol 2 - 3)',
        'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress)',
        'medical - seizure',
        'medical - st elevation myocardial infarction (acs/ami) (protocol 2 - 2)',
        'medical - stroke/tia (protocol 3 - 5)',
        'medical - supraventricular tachycardia (including atrial fibrillation) medical - tachycardia medical - ventricular tachycardia with a pulse',
        'ob/gyn - childbirth/labor/delivery (protocol 6 - 2)',
        'ob/gyn - pre-term labor (pre-term labor) (protocol 6 - 12)',
        'spinal motion restriction (protocol 4 - 13)'
    ]
    ungroup_p_node = [
        'airway - failed (protocol 3 - 16)',
        'environmental - heat exposure/heat exhaustion environmental - heat stroke (protocol 5 - 2)',
        'environmental - hypothermia (protocol 5 - 1)',
        'general - behavioral/patient restraint (protocol 3 - 4)',
        'general - cardiac arrest (protocol 2 - 7)',
        'general - cardiac arrest (protocol 8 - 2)',
        'general - pain control (protocol 3 - 10)',
        'general - pain control (protocol 9 - 8)',
        'hemorrhage control (protocol 4 - 14)',
        'injury - abdominal trauma (abdominal trauma) (protocol 4 - 2)',
        'injury - burns - thermal (protocol 10 - 2)',
        'injury - burns - thermal (protocol 4 - 3)',
        'injury - crush syndrome (protocol 4 - 4)',
        'injury - general trauma management (protocol 4 - 1)',
        'injury - head (protocol 10 - 4)',
        'injury - head (protocol 4 - 6)',
        'injury - thoracic (protocol 4 - 11)',
        'medical - abdominal pain (protocol 3 - 2)',
        'medical - allergic reaction/anaphylaxis (protocol 3 - 3)',
        'medical - allergic reaction/anaphylaxis (protocol 9 - 2)',
        'medical - altered mental status (protocol 3 - 15)',
        'medical - altered mental status (protocol 9 - 14)',
        'medical - bradycardia (protocol 2 - 9)',
        'medical - chest pain - cardiac suspected (protocol 2 - 1)',
        'medical - diabetic - hyperglycemia (protocol 3 - 7)',
        'medical - diabetic - hypoglycemia (protocol 3 - 8)',
        'medical - hypotension/shock (nontrauma) (protocol 3 - 13)',
        'medical - nausea/vomiting (protocol 3 - 9)',
        'medical - overdose/poisoning - calcium channel blocker (protocol 7 - 5)',
        'medical - overdose/poisoning - opioid (protocol 7 - 1)',
        'medical - overdose/poisoning - stimulant (protocol 7 - 2)',
        'medical - overdose/poisoning/toxic ingestion (protocol 9 - 9)',
        'medical - pulmonary edema/chf (protocol 2 - 3)',
        'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress - croup/epiglottits) (protocol 9 - 11)',
        'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress) (protocol 3 - 11)',
        'medical - seizure (protocol 3 - 12)',
        'medical - seizure (protocol 9 - 12)',
        'medical - st elevation myocardial infarction (acs/ami) (protocol 2 - 2)',
        'medical - stroke/tia (protocol 3 - 5)',
        'medical - supraventricular tachycardia (including atrial fibrillation) medical - tachycardia medical - ventricular tachycardia with a pulse (protocol 2 - 8)',
        'ob/gyn - childbirth/labor/delivery (protocol 6 - 2)',
        'ob/gyn - pre-term labor (pre-term labor) (protocol 6 - 12)',
        'spinal motion restriction (protocol 4 - 13)'
    ]
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
        'Medical - Supraventricular Tachycardia (including atrial fibrillation) Medical - Tachycardia Medical - Ventricular Tachycardia with a Pulse (Protocol 8 - 4)': 'Medical - Supraventricular Tachycardia (including atrial fibrillation) Medical - Tachycardia Medical - Ventricular Tachycardia with a Pulse',
        'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress) (protocol 3 - 11)': 'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress)',
        'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress - croup/epiglottits) (protocol 9 - 11)': 'medical - respiratory distress/asthma/copd/croup/reactive airway (respiratory distress)'
    }
    group_p_dict = {}
    for k, v in tmp_dict.items():
        group_p_dict[k.lower()] = v.lower()

    reverse_group_p_dict = defaultdict(list)
    for k, v in tmp_dict.items():
        reverse_group_p_dict[v.lower()].append(k.lower())

    #/home/xueren/Desktop/EMS
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'config_file/json files/multi-Graph/EMS/hier2p.json'), 'r') as f:
        hier2p = json.load(f)
    with open(os.path.join(dir_path, 'config_file/json files/multi-Graph/EMS/p2hier.json'), 'r') as f:
        p2hier = json.load(f)

    group_hier = []
    ungroup_hier = []
    group_hier_dict = {
        'adult cardiovascular emergencies': 'cardiovascular emergencies',
        'pediatric cardiovascular emergencies': 'cardiovascular emergencies',
        'adult general medical emergencies': 'medical emergencies',
        'pediatric medical emergencies': 'medical emergencies',
        'adult trauma emergencies': 'trauma emergencies',
        'pediatric trauma emergencies': 'trauma emergencies'
    }
    reverse_group_hier_dict = {
        'cardiovascular emergencies': ['adult cardiovascular emergencies',
                                       'pediatric cardiovascular emergencies'],
        'medical emergencies': ['adult general medical emergencies',
                                'pediatric medical emergencies'],
        'trauma emergencies': ['adult trauma emergencies',
                               'pediatric trauma emergencies']
    }
    for p in ungroup_p_node:
        cur_hier = p2hier[p].lower()
        if cur_hier not in ungroup_hier:
            ungroup_hier.append(cur_hier)
        ### group some hierarchy labels based on age
        if cur_hier == 'adult cardiovascular emergencies' or cur_hier == 'pediatric cardiovascular emergencies':
            cur_hier = 'cardiovascular emergencies'
        if cur_hier == 'adult general medical emergencies' or cur_hier == 'pediatric medical emergencies':
            cur_hier = 'medical emergencies'
        if cur_hier == 'adult trauma emergencies' or cur_hier == 'pediatric trauma emergencies':
            cur_hier = 'trauma emergencies'
        if cur_hier not in group_hier:
            group_hier.append(cur_hier)
    group_hier = sorted(group_hier)
    ungroup_hier = sorted(ungroup_hier)
elif dataset == 'MIMIC3':
    # MIMIC_3_DIR = '/home/xueren/Downloads/physionet.org/files/mimiciii/1.4'
    MIMIC_3_DIR = '/sfs/qumulo/qhome/zar8jw/EMS/data/MIMIC3'
    fname = '%s/ICD9_DIAG.json' % MIMIC_3_DIR
    with open(fname, 'r') as f:
        ICD9_DIAG = json.load(f)

    fname = '%s/ICD9_DIAG_GROUP.json' % MIMIC_3_DIR
    with open(fname, 'r') as f:
        ICD9_DIAG_GROUP = json.load(f)

    fname = '%s/group_ICD9.json' % MIMIC_3_DIR
    with open(fname, 'r') as f:
        group_ICD9_dict = json.load(f)

    fname = '%s/reverse_group_ICD9.json' % MIMIC_3_DIR
    with open(fname, 'r') as f:
        reverse_group_ICD9_dict = json.load(f)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(dir_path, 'json files/multi-Graph/MIMIC3/p2hier.json')
    with open(fname, 'r') as f:
        ICD92hier = json.load(f)
else:
    raise Exception('check dataset in default_sets.py')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True