import torch.nn as nn
import torch
import os
from EMS_Agent.Interface.default_sets import seed_everything, device, ungroup_p_node
import numpy as np
import warnings
import yaml
from EMS_Agent.Interface.utils import AttrDict, onehot2p, convert_label, preprocess
from EMS_Agent.Interface.default_sets import model_name
from EMS_Agent.Interface.Heterogeneous_graph import HeteroGraph
from EMS_Agent.Interface.model import EMSMultiModel
from transformers import BertTokenizer
from Feedback import FeedbackClient
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")
from classes import FeedbackObj, ProtocolObj
import time
import sys
from re import match
from classes import GUISignal

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot

from multiprocessing import Process

sys.path.append('../Demo')
import pipeline_config

class EMSTinyBERT(nn.Module):
    def __init__(self, config, date):
        super(EMSTinyBERT, self).__init__()
        self.config = config
        self.device = torch.device("cuda")

        self.tokenizer = BertTokenizer.from_pretrained(self.config.backbone, do_lower_Case=True, local_files_only=True)
        self.clean_model_date = date
        self.save_model_root = os.path.join('EMS_Agent/Interface/models', '{}'.format(self.clean_model_date))
        if self.config.graph == 'hetero':
            print('start reading file')
            start_t = time.time()
            signs_df = pd.read_excel('./EMS_Agent/Interface/config_file/All Protocols Mapping.xlsx')
            impre_df = pd.read_excel('./EMS_Agent/Interface/config_file/Impression Protocol.xlsx')
            med_df = pd.read_excel('./EMS_Agent/Interface/config_file/Medication Protocol.xlsx')
            proc_df = pd.read_excel('./EMS_Agent/Interface/config_file/Procedure Protocol.xlsx')
            end_t = time.time()
            print(f'time to load files: {end_t - start_t}')
            print('end reading files')


            print('start building graph')
            start_t = time.time()
            HGraph = HeteroGraph(backbone=self.config.backbone, mode=self.config.cluster)
            self.graph = HGraph(signs_df, impre_df, med_df, proc_df).to(self.device)
            end_t = time.time()
            print(f'time to build graph: {end_t - start_t}')
            print('end building graph')
        else:
            self.graph = None

        print('start building model')
        start_t = time.time()
        self.model = EMSMultiModel(self.config.backbone, self.config.max_len, self.config.attn,
                                   self.config.cluster, self.config.fusion, self.config.cls, self.config.graph)
        end_t = time.time()
        print(f'time to build model: {end_t - start_t}')
        print('end building model')


        model_path = os.path.join(self.save_model_root, 'model.pt')
        # checkpoint = torch.load(model_path, map_location='cpu')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint, strict=False)

        self.feedback_client = FeedbackClient()
        self.feedback_client.start()
        print("[Protocol Model: Device ]",self.device)
        self.model.to(self.device)

    def initData(self, text):
        ids, mask = None, None
        if text:
            text = preprocess(text)
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
            outputs = None
            if data['ids'] != None:
                ids = data["ids"].to(self.device, dtype=torch.long).unsqueeze(0)
                mask = data["mask"].to(self.device, dtype=torch.long).unsqueeze(0)
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

        # pred_protocol = np.array(ungroup_p_node)[np.where(preds == 1)]
        # pred_prob = logits[np.where(preds == 1)]

        # # get the protocol prediction with highest confidence value
        # max_prob_index = np.argmax(pred_prob)
        # prob = pred_prob[max_prob_index]
        # protocol = pred_protocol[max_prob_index]

        # # get the protocol prediction with top3
        indexes = np.argsort(logits)[::-1][:3]
        prob = np.array(logits)[indexes]
        protocol = np.array(ungroup_p_node)[indexes]
        print(indexes)
        print(prob)
        print(protocol)


        return protocol, prob, one_hot, logits


class EMSAgentInference(QThread):


    def __init__(self, Window, EMSTinyBERTQueue, ProtocolQueue):
        super().__init__()
        seed_everything(3407)
        # initialize

        self.window = Window
        self.speech_queue = EMSTinyBERTQueue
        self.protocol_queue = ProtocolQueue

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

        print('\n\n ================= Warmup Protocol Model =================')

        start_t = time.time()
        self.model = EMSTinyBERT(config, model_name)
        end_t = time.time()
        print(f"[Protocol model initialization time: {end_t-start_t}]")

        warmup_text = "hello"
        out = self.model(warmup_text)


        # end_t = time.time()

        # print(f"[Protocol model initialization time: {end_t-start_t}]")

        # print(f'[Protocol warm up done!: {model("Warmup Text")}]')

        #Signal warmup done
        protocolFB =  FeedbackObj("protocol model warmup done", "protocol model warmup done","protocol warmup done",  "protocol model warmup done")
        self.protocol_queue.put(protocolFB)


        print('================= Warmup Protocol Model Done =================\n\n')


        # with EMSTinyBERTQueue.mutex:
        #     EMSTinyBERTQueue.queue.clear()



    def run(self):

        if self.window:
            ProtocolSignal = GUISignal()
            ProtocolSignal.signal.connect(self.window.UpdateProtocolBoxes)
            

        narrative = ""

        # call the model    

        while True:
            # Get queue item from the Speech-to-Text Module
            print("[Protocol Model - Waiting for speech......]")
            received = self.speech_queue.get()


            # TODO: make thread exit while True loop based on threading module event
            if(received == 'Kill'):
                print("[Protocol Model Thread received reset signal. Resetting the narrative]")
                narrative = ""
                continue

            print('=============================================================')
            try:
                print(f'[Protocol model received transcript: {received.transcript}]')
                print(f'[Protocol model current narrative: {narrative}]')

                if received.transcript == "":
                    pred, prob, one_hot, logits = -1, -1, -1, -1
                    continue
                
                narrative = received.transcript

                start = time.perf_counter_ns()
                pred, prob, one_hot, logits = self.model(narrative)
                end = time.perf_counter_ns()
                print(f'[Protocol suggestion:{pred}:{prob}]')
                prot_latency = (end-start)/1000000

                #Send data (send to vision)

                protocolFB =  FeedbackObj("", pred[0], prob[0], "")
                protocolFeedback = ProtocolObj(pred[0],prob[0])
                protocol_dict = protocolFeedback.__dict__
                self.model.feedback_client.send_message(protocol_dict, 'protocolFeedback') 
                
                self.protocol_queue.put(protocolFB)

                top_3_protocols = []
                if self.window:
                    for prd, prb in zip(list(pred),list(prob)):
                        protocolFB =  FeedbackObj("", prd, prb, "")
                        top_3_protocols.append(protocolFB)
                    ProtocolSignal.signal.emit(top_3_protocols)


    # ===== save end to end pipeline results for this segment =========================================================================
                if pipeline_config.evaluation:
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
                    # print('logits', logits.shape)
                    pipeline_config.trial_data['logits'].append(str(logits))
                            
            except Exception as e:
                print("Error in Protocol Model", e)
                continue
