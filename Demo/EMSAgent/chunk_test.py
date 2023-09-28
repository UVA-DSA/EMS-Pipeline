from transformers import BertTokenizer
import pandas as pd
import os
from EMSAgent.utils import text_remove_double_space, AttrDict, onehot2p
from EMSAgent.default_sets import group_p_dict, device, p_node, reverse_group_p_dict, seed_everything
import torch
import torch.nn as nn
from EMSAgent.Heterogeneous_graph import HeteroGraph
from EMSAgent.model import EMSMultiModel
import yaml
import re
import numpy as np


class CHUNKModel(nn.Module):
    def __init__(self, config, date):
        super(CHUNKModel, self).__init__()
        self.config = config
        self.p_node = p_node
        self.tokenizer = BertTokenizer.from_pretrained(self.config.backbone, do_lower_Case=True)
        self.clean_model_date = date
        self.save_model_root = os.path.join('./models', '{}'.format(self.clean_model_date),
                                            'iter:{}_lr:{}_bs:{}_epoch:{}_lossfn:{}_backbone:{}_graph:{}_cluster:{}_cls:{}_attn:{}'.format(
                                                0, self.config.learning_rate, self.config.batch_size,
                                                self.config.epochs, self.config.loss_func, self.config.backbone,
                                                self.config.graph, self.config.cluster, self.config.cls, self.config.attn))
        if self.config.graph == 'hetero':
            signs_df = pd.read_excel('/home/xueren/Desktop/EMS/data/EMS/Protocol_Impression files/All Protocols Mapping.xlsx')
            impre_df = pd.read_excel('/home/xueren/Desktop/EMS/data/EMS/Protocol_Impression files/Impression Protocol.xlsx')
            med_df = pd.read_excel('/home/xueren/Desktop/EMS/data/EMS/Protocol_Impression files/Medication Protocol.xlsx')
            proc_df = pd.read_excel('/home/xueren/Desktop/EMS/data/EMS/Protocol_Impression files/Procedure Protocol.xlsx')
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

    @staticmethod
    def removePunctuation(sentence):
        sentence = re.sub(r'[?|!|\'|"|;|:|#]', r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)
        sentence = sentence.strip()
        sentence = sentence.replace("\n", " ")
        return sentence

    @staticmethod
    def removechar(sentence):
        text = ""
        for word in sentence.split():
            if len(word) > 1:
                text += word + ' '
        return text

    def initData(self, text):
        text = self.removePunctuation(text).lower()
        text = self.removechar(text).lower()
        inputs = self.tokenizer.__call__(text,
                                          None,
                                          add_special_tokens=True,
                                          max_length=self.config.max_len,
                                          padding="max_length",
                                          truncation=True,
                                          )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }

    def eval_fn(self, data):
        with torch.no_grad():
            ids = data["ids"].to(device, dtype=torch.long).unsqueeze(0)
            mask = data["mask"].to(device, dtype=torch.long).unsqueeze(0)
            outputs, feats, meta_feats, graph_feats = self.model(ids=ids, mask=mask,
                                                    meta_kn_ids=None, meta_kn_mask=None,
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


    def forward(self, texts):
        input = self.initData(texts)
        preds, logits = self.eval_fn(input)
        logits = torch.softmax(logits, dim=-1)

        logits = logits.cpu().numpy()[0]
        preds = preds.cpu().numpy().item()
        return preds, logits


def chunkPipeline(config, date):
    haydon = pd.read_excel('/home/xueren/Desktop/EMS/data/EMS/haydon chunks.xlsx', sheet_name=None)
    model = CHUNKModel(config, date)
    all_preds = []
    all_probs = []
    for name, sheet in haydon.items():
        prediction = []
        prediction_prob = []
        label = sheet['Protocols'][0].strip().lower()
        for i in range(len(sheet)):
            if type(sheet['combined_transcription_so_far'][i]) != float:
                text = sheet['combined_transcription_so_far'][i]
            else:
                text = 'unknown'

            preds, logits = model(text)
            prob = logits[preds]
            preds = p_node[preds]
            prediction.append(preds)
            prediction_prob.append(prob)
        all_preds.append(prediction)
        all_probs.append(prediction_prob)
        sheet.insert(6, 'pred', prediction)
        sheet.insert(7, 'probability', prediction_prob)
        # think about what to do with all_preds #
    return haydon


if __name__ == '__main__':
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
    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=loader)
    config['parameters'].update({'iter_num': {'value': 0}})
    config = AttrDict(config['parameters'])
    date = '2023-03-01-09_52_18' #[2023-03-01-09_52_18, 2023-03-01-10_06_45]
    haydon_df = chunkPipeline(config, date)
    writer = pd.ExcelWriter('./haydon.xlsx', engine='openpyxl')
    for key, val in haydon_df.items():
        val.to_excel(writer, sheet_name=key)
    writer.save()