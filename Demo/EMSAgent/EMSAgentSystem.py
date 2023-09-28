import json
import torch.nn as nn
import torch
import os
from EMSAgent.default_sets import seed_everything, device, p_node
import numpy as np
import warnings
import yaml
import re
from EMSAgent.utils import AttrDict, onehot2p
from EMSAgent.Heterogeneous_graph import HeteroGraph
from EMSAgent.model import EMSMultiModel
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")
from classes import  GUISignal
import time


# ------------ For Feedback ------------
class FeedbackObj:
    def __init__(self, intervention, protocol, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.concept = concept

class EMSAgent(nn.Module):
    def __init__(self, config, date):
        super(EMSAgent, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.backbone, do_lower_Case=True)
        self.clean_model_date = date
        self.save_model_root = os.path.join('../../EMSAgent/Interface/models', '{}'.format(self.clean_model_date))
        if self.config.graph == 'hetero':
            signs_df = pd.read_excel('./EMSAgent/config_file/All Protocols Mapping.xlsx')
            impre_df = pd.read_excel('./EMSAgent/config_file/Impression Protocol.xlsx')
            med_df = pd.read_excel('./EMSAgent/config_file/Medication Protocol.xlsx')
            proc_df = pd.read_excel('./EMSAgent/config_file/Procedure Protocol.xlsx')
            HGraph = HeteroGraph(backbone=self.config.backbone, mode=self.config.cluster)
            self.graph = HGraph(signs_df, impre_df, med_df, proc_df)
        else:
            self.graph = None
        self.model = EMSMultiModel(self.config.backbone, self.config.max_len, self.config.attn,
                                   self.config.cluster, self.config.fusion, self.config.cls, self.config.graph)

        model_path = os.path.join(self.save_model_root, 'model.pt')
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.to(device)

    def initData(self, text):
        ids, mask = None, None
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


        return {'ids': ids, 'mask': mask}

    def eval_fn(self, data):
        with torch.no_grad():
            if data['ids'] != None:
                ids = data["ids"].to(device, dtype=torch.long).unsqueeze(0)
                mask = data["mask"].to(device, dtype=torch.long).unsqueeze(0)
                outputs, text_feats, graph_feats = self.model(ids=ids, mask=mask, G=self.graph)


            # if it's multi-class classification
            # outputs = torch.softmax(outputs, dim=-1)
            # _, preds = torch.max(outputs, dim=1)

            # if it's multi-label classification
            outputs = torch.sigmoid(outputs).squeeze()
            preds = np.where(outputs.cpu().numpy() > 0.5, 1, 0)
            if not preds.any():
                idx = torch.argmax(outputs).cpu().numpy()
                preds[idx] = 1


        return preds, outputs

    def forward(self, text):
        input = self.initData(text)
        preds, logits = self.eval_fn(input)

        ### multi-class classification
        # logits = logits.cpu().numpy()[0]
        # pred_protocol = p_node[preds]
        # pred_prob = logits[preds]

        ### multi-label classification
        logits = logits.cpu().numpy()
        pred_protocol = np.array(p_node)[np.where(preds == 1)]
        pred_prob = logits[np.where(preds == 1)]

        return pred_protocol, pred_prob


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
    
    root = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(root, 'config.yaml'), 'r') as f:
        config = yaml.load(f, Loader=loader)
    config = AttrDict(config['parameters'])
    from EMSAgent.default_sets import date
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
                    print("Received chunk", received.transcript, str.isspace(received.transcript))
                    if(not str.isspace(received.transcript)):

                        narrative += received.transcript

                        try:
                                
                            start = time.time()
                            
                            pred, prob = model(narrative)
                            pred = ','.join(pred)
                            prob = ','.join(str(p) for p in prob)
                            print(pred, prob)
                            end = time.time()
                            ProtocolSignal.signal.emit(["(Protocol: " +str(pred) + " : " +str(prob) +")"])
                            print('executation time: {:.4f}'.format(end - start))

                            #Feedback
                            protocolFB =  FeedbackObj(None, str(pred) + " : " +str(round(prob,2)), None)
                            FeedbackQueue.put(protocolFB)


                            #write data to file for data collection
                            f.write(narrative+" : ")
                            f.write(str(pred))
                            f.write(str(prob))
                            f.write("\n")
                        except:
                            print("Protocol Prediction Failure!")
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
                print("Received chunk", received.transcript, str.isspace(received.transcript))
                if(not str.isspace(received.transcript)):
                    narrative += received.transcript
                    try:
                        start = time.time()
                        pred, prob = model(narrative)
                        end = time.time()
                        # convert list to string
                        pred = ','.join(pred)
                        prob = ','.join(str(p) for p in prob)
                        ProtocolSignal.signal.emit(["(Protocol: " +str(pred) + " : " +str(prob) +")"])
                        print('executation time: {:.4f}'.format(end - start))
                        print(pred, prob)

                        #Feedback
                        protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                        FeedbackQueue.put(protocolFB)

                    except:
                        print("Protocol Prediction Failure!")