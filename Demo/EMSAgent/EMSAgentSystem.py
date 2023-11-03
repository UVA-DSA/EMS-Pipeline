import torch.nn as nn
import torch
import os
from EMSAgent.default_sets import seed_everything, device, ungroup_p_node
import numpy as np
import warnings
import yaml
import re
from EMSAgent.utils import AttrDict, onehot2p, convert_label
from EMSAgent.Heterogeneous_graph import HeteroGraph
from EMSAgent.model import EMSMultiModel
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")
from classes import  FeedbackObj
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

        one_hot = preds
        ### multi-class classification
        # logits = logits.cpu().numpy()[0]
        # pred_protocol = p_node[preds]
        # pred_prob = logits[preds]

        ### multi-label classification
        logits = logits.cpu().numpy()
        if self.config.cluster == 'group':
            preds, logits = convert_label([preds], ages=[55], logits=[logits])
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

    try:
        os.mkfifo(pipeline_config.protocol_fifo)
        print("[Protocol Pipe Created!]")
    except FileExistsError:
        print("[Protocol Pipe Exists!]")
        
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
    protocolFB =  FeedbackObj("wd", "wd", "wd")
    FeedbackQueue.put(protocolFB)

    if(pipeline_config.action_recognition):
        try:
            with open(pipeline_config.protocol_fifo, "w") as fifo:

                # call the model    
                while True:
                    # Get queue item from the Speech-to-Text Module+
                    received = EMSAgentQueue.get()

                    # TODO: make thread exit while True loop based on threading module event
                    if(received == 'Kill'):
                        # print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
                        break
                    else:
                        print('=============================================================')
                        print(f'[Protocol model received transcript: {received.transcript}]')
                        # initialize variables
                        # try:
                        start = time.perf_counter()
                        pred, prob, one_hot, logits = model(received.transcript)
                        end = time.perf_counter()
                        print(f'[Protocol suggestion:{pred}:{prob}]')
                        prot_latency = (end-start)*1000

                        #Feedback
                        protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                        FeedbackQueue.put(protocolFB)

                        try:
                            fpred = pred.split(',')[0]
                            fifo.write(fpred+'\n')
                            fifo.flush()
                        except Exception as e:
                            print(f'[Protocol Pipe Writing Failed:{str(e)}]')

            # ===== save end to end pipeline results for this segment =========================================================================
                        '''
                        Fields with placeholders are calculated later in EndToEndEval or another data processing script
                        '''
                        pipeline_config.trial_data['speech latency (ms)'].append(received.transcriptionDuration)
                        pipeline_config.trial_data['transcript'].append(received.transcript)
                        pipeline_config.trial_data['whisper confidence'].append(received.confidence)
                        pipeline_config.trial_data['WER'].append(0) #placeholder
                        pipeline_config.trial_data['CER'].append(0) #placeholder
                        pipeline_config.trial_data['protocol latency (ms)'].append(prot_latency)
                        pipeline_config.trial_data['protocol prediction'].append(pred)
                        pipeline_config.trial_data['protocol confidence'].append(prob)
                        pipeline_config.trial_data['protocol correct?'].append(0) #placeholder
                        pipeline_config.trial_data['one hot prediction'].append(str(one_hot))
                        pipeline_config.trial_data['one hot gt'].append(str('[one hot gt placeholder]'))
                        pipeline_config.trial_data['tn'].append('tn placeholder')
                        pipeline_config.trial_data['fp'].append('fp placeholder')
                        pipeline_config.trial_data['fn'].append('fn placeholder')
                        pipeline_config.trial_data['tp'].append('tp placeholder')
                        pipeline_config.trial_data['logits'].append(str(logits))


        except Exception as e:
            print("[EMSAgent Exception] ",str(e))      
                    
    else:
        try:
            # call the model    
            while True:
                # Get queue item from the Speech-to-Text Module+
                received = EMSAgentQueue.get()

                # TODO: make thread exit while True loop based on threading module event
                if(received == 'Kill'):
                    # print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
                    break
                else:
                    print('=============================================================')
                    print(f'[Protocol model received transcript: {received.transcript}]')
                    # try:
                    start = time.perf_counter()
                    pred, prob, one_hot, logits = model(received.transcript)
                    end = time.perf_counter()
                    print(f'[Protocol suggestion:{pred}:{prob}]')
                    prot_latency = (end-start)*1000


                    #Feedback
                    protocolFB =  FeedbackObj("", str(pred) + " : " +str(prob), "")
                    FeedbackQueue.put(protocolFB)

        # ===== save end to end pipeline results for this segment =========================================================================
                    '''
                    Fields with placeholders are calculated later in EndToEndEval or another data processing script
                    '''
                    pipeline_config.trial_data['speech latency (ms)'].append(received.transcriptionDuration)
                    pipeline_config.trial_data['transcript'].append(received.transcript)
                    pipeline_config.trial_data['whisper confidence'].append(received.confidence)
                    pipeline_config.trial_data['WER'].append(0) #placeholder
                    pipeline_config.trial_data['CER'].append(0) #placeholder
                    pipeline_config.trial_data['protocol latency (ms)'].append(prot_latency)
                    pipeline_config.trial_data['protocol prediction'].append(pred)
                    pipeline_config.trial_data['protocol confidence'].append(prob)
                    pipeline_config.trial_data['protocol correct?'].append(0) #placeholder
                    pipeline_config.trial_data['one hot prediction'].append(str(one_hot))
                    pipeline_config.trial_data['one hot gt'].append(str('[one hot gt placeholder]'))
                    pipeline_config.trial_data['tn'].append('tn placeholder')
                    pipeline_config.trial_data['fp'].append('fp placeholder')
                    pipeline_config.trial_data['fn'].append('fn placeholder')
                    pipeline_config.trial_data['tp'].append('tp placeholder')
                    pipeline_config.trial_data['logits'].append(str(logits))

        except Exception as e:
            print("[EMSAgent Exception] ",str(e))            