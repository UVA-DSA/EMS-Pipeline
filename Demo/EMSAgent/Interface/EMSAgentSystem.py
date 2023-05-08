import json
import torch.nn as nn
import torch
import os
from .default_sets import seed_everything, device, p_node
import numpy as np
import warnings
import yaml
import re
from .utils import AttrDict, onehot2p
from .Heterogeneous_graph import HeteroGraph
from .model import EMSMultiModel
from transformers import BertTokenizer
import pandas as pd
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")
from classes import  GUISignal

# ------------ For Feedback ------------
class FeedbackObj:
    def __init__(self, intervention, protocol, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.concept = concept

# ------------ End Feedback Obj Class ------------

class EMSAgent(nn.Module):
    def __init__(self, config, date):
        super(EMSAgent, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.backbone, do_lower_Case=True)
        self.clean_model_date = date
        self.save_model_root = os.path.join('./EMSAgent/Interface/models', '{}'.format(self.clean_model_date))
        if self.config.graph == 'hetero':
            signs_df = pd.read_excel('./EMSAgent/Interface/config_file/All Protocols Mapping.xlsx')
            impre_df = pd.read_excel('./EMSAgent/Interface/config_file/Impression Protocol.xlsx')
            med_df = pd.read_excel('./EMSAgent/Interface/config_file/Medication Protocol.xlsx')
            proc_df = pd.read_excel('./EMSAgent/Interface/config_file/Procedure Protocol.xlsx')
            HGraph = HeteroGraph(backbone=self.config.backbone, mode=self.config.cluster)
            self.graph = HGraph(signs_df, impre_df, med_df, proc_df)
        else:
            self.graph = None
        self.model = EMSMultiModel(self.config.backbone, self.config.dropout, self.config.max_len, self.config.attn,
                                   self.config.cluster, self.config.cls, self.graph)


        model_path = os.path.join(self.save_model_root, 'model.pt')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model.to(device)

    def initData(self, text, meta_kn=None):
        ids, mask = None, None
        meta_ids, meta_mask = None, None
        if text:
            inputs = self.tokenizer.__call__(text,
                                              None,
                                              add_special_tokens=True,
                                              max_length=self.config.max_len,
                                              padding="max_length",
                                              truncation=True,
                                              )
            ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
            mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        if meta_kn:
            meta_inputs = self.tokenizer.__call__(meta_kn,
                                                  None,
                                                  add_special_tokens=True,
                                                  max_length=self.config.max_len,
                                                  padding="max_length",
                                                  truncation=True,
                                                  )
            meta_ids = torch.tensor(meta_inputs["input_ids"], dtype=torch.long)
            meta_mask = torch.tensor(meta_inputs["attention_mask"], dtype=torch.long)

        return {'ids': ids, 'mask': mask, 'meta_ids': meta_ids, 'meta_mask': meta_mask}

    def eval_fn(self, data):
        with torch.no_grad():
            if data['ids'] != None:
                ids = data["ids"].to(device, dtype=torch.long).unsqueeze(0)
                mask = data["mask"].to(device, dtype=torch.long).unsqueeze(0)
                outputs, feats, meta_feats, graph_feats = self.model(ids=ids, mask=mask,
                                                        meta_kn_ids=None, meta_kn_mask=None,
                                                        G=self.graph)

            if data['meta_ids'] != None:
                meta_ids = data["meta_ids"].to(device, dtype=torch.long).unsqueeze(0)
                meta_mask = data["meta_mask"].to(device, dtype=torch.long).unsqueeze(0)
                outputs, feats, meta_feats, graph_feats = self.model(ids=None, mask=None,
                                                        meta_kn_ids=meta_ids, meta_kn_mask=meta_mask,
                                                        G=self.graph)

            # if it's multi-class classification
            _, preds = torch.max(outputs, dim=1)

            # # if it's multi-label classification
            # outputs = torch.sigmoid(outputs).squeeze()
            # preds = np.where(outputs.cpu().numpy() > 0.5, 1, 0)
            # if not preds.any():
            #     idx = torch.argmax(outputs).cpu().numpy()
            #     preds[idx] = 1


        return preds, outputs

    def forward(self, text, meta_kn=None):
        input = self.initData(text, meta_kn)
        preds, logits, = self.eval_fn(input)
        logits = torch.softmax(logits, dim=-1)

        logits = logits.cpu().numpy()[0]
        preds = preds.cpu().numpy().item()
        pred_protocol = p_node[preds]
        return pred_protocol, logits[preds]


def EMSAgentSystem(Window, EMSAgentSpeechToNLPQueue, FeedbackQueue, data_path_str, protocolStreamBool):


    ProtocolSignal = GUISignal()
    ProtocolSignal.signal.connect(Window.UpdateProtocolBoxes)

    # initialize
    seed_everything(3407)
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open('./EMSAgent/Interface/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=loader)
    config['parameters'].update({'iter_num': {'value': 0}})
    config = AttrDict(config['parameters'])
    date = '2023-03-01-09_52_18' #[2023-03-01-09_52_18, 2023-03-01-10_06_45]
    model = EMSAgent(config, date)

    # call the model
    # narrative = """ATF 64 yom unresponsive in bathroom, delay in accessing patient due to bathroom door being locked with no key.Patient family states he went to the bathroom at appx 2000 hrs, they realized that they hadn't seen him in while, so they checked on him, heard him snoring, and he wouldn't respond, so they called us. Upon our arrival it was appx 2115 hrs. Patient had paraphenalia laying beside him. Patient has history of heroin use.ATF 64 yom laying in bathroom. Patient unresponsive with snoring type respirations, spoon and needle laying next to him. Remainder of assessment held until after given. Patient AAOx4, ABC's intact, GCS 15. Patient admits to heroin use, states he has been off for the last couple years, but has relapsed for the last three day. Patient pupils 2mm RTL, skin normal, warm, and dry. Patient has symmetrical facial features with no slurring or droop noted. Patient has no obvious trauma, but has dirt and wood chips to forehead from laying on bathroom floor. Patient reports to have headache 9/10 to entire head. Patient has no signs of respiratory distress, no JVD. Lung sound clear. Patient has no chest pain, no abdominal pain, no shortness of breath. Patient has good PMS to all extremities, but does report that he has some pain to both knees from where he fell yesterday (he tripped over rug)"""
    narrative = ""

    if protocolStreamBool == True:
        if not os.path.exists(data_path_str + "protocoldata_XuerenModel/"):
            os.makedirs(data_path_str + "protocoldata_XuerenModel/")
        
        with open(data_path_str + "protocoldata_XuerenModel/emsagentlog.txt", 'w') as f:
            while True:

                # Get queue item from the Speech-to-Text Module
                received = EMSAgentSpeechToNLPQueue.get()

                if(received == 'Kill'):
                    # print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
                    break

                if(Window.reset == 1):
                    # print("Cognitive System Thread Received reset signal. Killing Cognitive System Thread.")
                    return

                # If item received from queue is legitmate
                else:
                    print("Received chunk", received.transcript)
                    narrative += received.transcript
                    start = time.time()
                    pred, prob = model(narrative)
                    end = time.time()
                    ProtocolSignal.signal.emit(["(Xueren Model: " +str(pred) + " : " +str(prob) +")"])
                    print('executation time: {:.4f}'.format(end - start))
                    print(pred, prob)

                    #Feedback
                    protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                    FeedbackQueue.put(protocolFB)


                    #write data to file for data collection
                    f.write(narrative+" : ")
                    f.write(str(pred))
                    f.write(str(prob))
                    f.write("\n")
                    
            f.close()
            
    if protocolStreamBool == False:
        while True:
            # Get queue item from the Speech-to-Text Module
            received = EMSAgentSpeechToNLPQueue.get()

            if(received == 'Kill'):
                # print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
                break

            if(Window.reset == 1):
                # print("Cognitive System Thread Received reset signal. Killing Cognitive System Thread.")
                return

            # If item received from queue is legitmate
            else:
                print("Received chunk", received.transcript)
                narrative += received.transcript
                start = time.time()
                pred, prob = model(narrative)
                end = time.time()
                ProtocolSignal.signal.emit(["(Xueren Model: " +str(pred) + " : " +str(prob) +")"])
                print('executation time: {:.4f}'.format(end - start))
                print(pred, prob)

                #Feedback
                protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                FeedbackQueue.put(protocolFB)
            
                


