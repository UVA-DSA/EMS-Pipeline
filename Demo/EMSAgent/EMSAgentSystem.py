import torch.nn as nn
import torch
import os
from EMSAgent.default_sets import seed_everything, device, ungroup_p_node
import numpy as np
import warnings
import yaml
from EMSAgent.utils import AttrDict, onehot2p, convert_label
from EMSAgent.Heterogeneous_graph import HeteroGraph
from EMSAgent.model import EMSMultiModel
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")
from classes import   FeedbackObj
import time
import sys
from re import match

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
        input = self.initData(text)
        preds, logits = self.eval_fn(input)
        one_hot = preds
        ### multi-class classification
        # logits = logits.cpu().numpy()[0]
        # pred_protocol = p_node[preds]
        # pred_prob = logits[preds]

        # find age
        regex = '[0-9]+ (y.?r.?s?|years? old)'
        output = match(regex, text)
        if output:
            age_s, _ = output.group().split('y')
            age = int(age_s)
        else:
            age = 39 #assume adult if age not given - 39 is median age for adults in US

        ### multi-label classification
        logits = logits.cpu().numpy()
        if self.config.cluster == 'group':
            preds, logits = convert_label([preds], ages=[age], logits=[logits])
            preds, logits = preds[0], logits[0]
            one_hot = preds

        pred_protocol = np.array(ungroup_p_node)[np.where(preds == 1)]
        pred_prob = logits[np.where(preds == 1)]

        # get the protocol prediction with highest confidence value
        max_prob_index = np.argmax(pred_prob)
        prob = pred_prob[max_prob_index]
        protocol = pred_protocol[max_prob_index]

        return protocol, prob, one_hot, logits


def EMSAgentSystem(EMSAgentQueue, FeedbackQueue):

    # ProtocolSignal.signal.connect(Window.UpdateProtocolBoxes)
    # initialize
    seed_everything(3407)
    from EMSAgent.default_sets import model_name

    config = {
        'max_len': 512,
        'fusion': None,
        'cls': 'fc'
    }
    if model_name == 'EMSAssist':
        config['backbone'] = 'google/mobilebert-uncased'
        config['cluster'] = 'ungroup'
        config['attn'] = None
        config['graph'] = None
    elif model_name == 'DKEC-TinyClinicalBERT':
        config['backbone'] = 'nlpie/tiny-clinicalbert'
        config['cluster'] = 'group'
        config['attn'] = 'la'
        config['graph'] = 'hetero'
    else:
        raise Exception('wrong model name')
    config = AttrDict(config)


    model = EMSAgent(config, model_name)
    
    print('================= Warmup Protocol Model =================')
    print(f'[Protocol warm up done!: {model("Warmup Text")}]')

    #Signal warmup done
    protocolFB =  FeedbackObj("protocol model warmup done", "protocol model warmup done", "protocol model warmup done")
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
            if len(received.transcript) and not received.transcript.isspace():
                # try:
                start = time.perf_counter_ns()
                pred, prob, one_hot, logits = model(received.transcript)
                end = time.perf_counter_ns()
                ProtocolSignal.signal.emit([f"(Protocol:{pred}:{prob})"])
                print(f'[Protocol suggestion:{pred}:{prob}]')

                #Feedback
                protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                FeedbackQueue.put(protocolFB)

            else:
                pred = 'Protocol is not suggested due to receiving blank space as transcript'
                print(f'[{pred}]')

 # ===== save end to end pipeline results for this segment =========================================================================
            # 'wer' and 'cer' calcluated and replaced later in EndToEndEval.py
            pipeline_config.curr_segment += [received.transcriptionDuration, received.transcript, received.confidence, 'wer', 'cer']
            # if we made a protocol prediction
            if start != None and end != None:
                pipeline_config.curr_segment += [(end-start)/1000000, pred, prob, 'correct?', one_hot, 'one hot GT', 'tn', 'fp', 'fn', 'tp', logits] # see if protocol prediction is correct later in EndToEndEval.py
            else:
                # if no suggesion, save one hot vector of all 0's
                pipeline_config.curr_segment += [-1, pred, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            pipeline_config.rows_trial.append(pipeline_config.curr_segment)
            pipeline_config.curr_segment = []

                
        
