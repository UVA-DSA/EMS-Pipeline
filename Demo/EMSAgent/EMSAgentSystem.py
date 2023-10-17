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
from classes import   FeedbackObj
# from classes import  GUISignal, FeedbackObj
import time
import sys

sys.path.append('../Demo')
import pipeline_config

class EMSAgent(nn.Module):
    def __init__(self, config, date):
        super(EMSAgent, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.backbone, do_lower_Case=True)
        self.clean_model_date = date
        self.save_model_root = os.path.join('EMSAgent/models', '{}'.format(self.clean_model_date))
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
        self.model.load_state_dict(checkpoint, strict=False)
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

        start = time.perf_counter()
        input = self.initData(text)
        end = time.perf_counter()
        print(f"Protocol: Tokenization Latency: {end-start}")

        start = time.perf_counter()
        preds, logits = self.eval_fn(input)
        end = time.perf_counter()
        print(f"Protocol: Prediction Latency: {end-start}")

        ### multi-class classification
        # logits = logits.cpu().numpy()[0]
        # pred_protocol = p_node[preds]
        # pred_prob = logits[preds]

        ### multi-label classification
        logits = logits.cpu().numpy()
        pred_protocol = np.array(p_node)[np.where(preds == 1)]
        pred_prob = logits[np.where(preds == 1)]

        return pred_protocol, pred_prob


def EMSAgentSystem(EMSAgentQueue, FeedbackQueue):

    # ProtocolSignal = GUISignal()

    # ProtocolSignal.signal.connect(Window.UpdateProtocolBoxes)

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

    print('================= Warmup Protocol Model =================')
    print(f'[Protocol warm up done!: {model("Warmup Text")}]')

    #Signal warmup done
    protocolFB =  FeedbackObj("wd", "wd", "wd")
    FeedbackQueue.put(protocolFB)
    
    # call the model    
    while True:
        # Get queue item from the Speech-to-Text Module
        received = EMSAgentQueue.get()

        # TODO: make thread exit while True loop based on threading module event
        if(received == 'Kill'):
            # print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
            break
        else:
            print('=============================================================')
            print(f'[Protocol model received transcript: {received.transcript}]')
            # initialize variables
            start = None
            end = None
            pred = None
            prob = None
            # received.transcript = "55 year old male found unconscious driver side passenger seat of his car his wife reported that he snorted a line of heroin before just prior to losing to consciousness patient originally presented unresponsive and pale with shallow ineffective respirations at a rate of about 5 with heart rate was 118 his blood pressure was 205 over 119 his blood glucose levels 126 his O2 saturations was were 94% patient required bag mask ventilation with with attached oxygen however after 0.25 mg of naloxone intravenously patient is now awake and breathing normally with Improvement in Vital Signs and respiratory status and no longer needs supplemental oxygen"
            # received.transcript = "55 year old male found unconscious driver "

            if len(received.transcript):
                try:
                    start = time.perf_counter()
                    pred, prob = model(received.transcript)
                    end = time.perf_counter()
                    pred = ','.join(pred)
                    prob = ','.join(str(p) for p in prob)
                    # ProtocolSignal.signal.emit([f"(Protocol:{pred}:{prob})"])
                    print(f'[Latency: {(end-start)}Protocol suggestion:{pred}:{prob}]')

                    #Feedback
                    protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                    FeedbackQueue.put(protocolFB)

                except Exception as e:
                    pred = 'Protocol is not suggested due to exception'
                    print(e, f'[{pred}]')
            else:
                pred = 'Protocol is not suggested due to receiving blank space as transcript'
                print(f'[{pred}]')

 # ===== save end to end pipeline results for this segment =========================================================================
            # 'wer' and 'cer' calcluated and replaced later in EndToEndEval.py
            pipeline_config.curr_segment += [received.transcriptionDuration, received.transcript, received.confidence, 'wer', 'cer']
            # if we never made a protocol prediction
            if start != None and end != None:
                pipeline_config.curr_segment += [end-start, pred, prob, 'correct?'] # see if protocol prediction is correct later in EndToEndEval.py
            else:
                pipeline_config.curr_segment += [-1, pred, -1, -1]
            pipeline_config.rows_trial.append(pipeline_config.curr_segment)
            pipeline_config.curr_segment = []

                
        
