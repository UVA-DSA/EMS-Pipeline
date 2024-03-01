# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tqdm import tqdm

from tensorflow_asr.metrics.error_rates import ErrorRate
from tensorflow_asr.utils.file_util import read_file
from tensorflow_asr.utils.metric_util import cer, wer

logger = tf.get_logger()
#devices = [2]
#gpus = tf.config.list_physical_devices("GPU")
#visible_gpus = [gpus[i] for i in devices]
#tf.config.set_visible_devices(visible_gpus, "GPU")
#strategy = tf.distribute.MirroredStrategy()

import fire
#import tensorflow as tf
from tensorflow_asr.utils.file_util import read_file
import json
from pydub import AudioSegment

from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio

def writeListFile(file_path, output_list, encoding = None):
    f = open(file_path, mode = "w")
    output_str = "\n".join(output_list)
    f.write(output_str)
    f.close()

def write2DListFile(file_path, output_list, line_sep = " "):
    str_list = []
    for out_line in output_list:
        str_line = []
        for e in out_line:
            str_line.append(str(e))
        str_list.append(str_line)
    out_list = list(map(line_sep.join, str_list))
    writeListFile(file_path, out_list)

def readFile(file_path, encoding = None):
    f = open(file_path, 'r')
    lines = f.read().splitlines()
#    print(lines[0])
    res = []
    for line in lines:
        line = line.strip()
        res.append(line)
    f.close()
    return res

def main(
#    filename: str = "/home/liuyi/TensorFlowASR/dataset/LibriSpeech/test-clean/5639/40744/5639-40744-0008.flac",
#    filename: str = "/home/liuyi/audio_data/radu0.m4a",
    tflite: str = "./tflite_models/pretrained_librispeech_train_ss_test_concatenated_epoch50_noOptimize.tflite", # +++++ .tflite model
#    tflite: str = "./tflite_models/subsampling-conformer.latest.tflite",
#    tflite: str = "",
    blank: int = 0,
    num_rnns: int = 1,
    nstates: int = 2,
    statesize: int = 320,
):

    concatenated_ss_audios_path = "/home/liuyi/audio_data/sample100/signs_symptoms_audio_concatenated" # +++++ audio path
    true_text_path = "/home/liuyi/tflite_experimental/emsBert/eval_pretrain/fitted_desc_sampled100e2e_test.tsv"

    true_lines = readFile(true_text_path)
    true_lines = true_lines[1:]

    # this is for computing wer of conformer.tflite 
    asr_true_text_path = "/home/liuyi/TensorFlowASR/conformer_standalone/finetune-test_concatenated_ss_transcripts.tsv"
    asr_true_lines = readFile(asr_true_text_path)
    asr_true_lines = asr_true_lines[1:]


#    with read_file(true_text_path) as path:
#        with open(path, "r", encoding="utf-8") as openfile:
#            true_lines = openfile.read().splitlines()
#            true_lines = true_lines[1:]  # skip header    
    transcribed_e2e_lines = []
    asr_wer_lines = []
    tflitemodel = tf.lite.Interpreter(model_path=tflite)    
    input_details = tflitemodel.get_input_details()
    output_details = tflitemodel.get_output_details()
    

    for idx, (true_line, asr_true_line) in enumerate(tqdm(zip(true_lines, asr_true_lines))):
#    idx = 0
#    true_line = true_lines[0]
#    print(true_line)
#    asr_true_line = asr_true_lines[0]
        audio_f = "sss" + str(idx+1) + ".wav"
        filename = os.path.join(concatenated_ss_audios_path, audio_f)
#    print(file_name)
    
#    with strategy.scope():
                
        signal = read_raw_audio(filename)
        tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
        tflitemodel.allocate_tensors()
        tflitemodel.set_tensor(input_details[0]["index"], signal)
        tflitemodel.set_tensor(input_details[1]["index"], tf.constant(blank, dtype=tf.int32))
        tflitemodel.set_tensor(input_details[2]["index"], tf.zeros([num_rnns, nstates, 1, statesize], dtype=tf.float32))
        tflitemodel.invoke()
        hyp0 = tflitemodel.get_tensor(output_details[0]["index"])
        hyp1 = tflitemodel.get_tensor(output_details[1]["index"])
        hyp2 = tflitemodel.get_tensor(output_details[2]["index"])
        
        # print("".join([chr(u) for u in hyp0]))
    
    
        true_label = true_line.split("\t")[1]
        pred_text = "".join([chr(u) for u in hyp0])
#        transcribed_e2e_lines.append([pred_text, true_label])
    
        asr_line_record = asr_true_line.split("\t")
    #    print("true text: ", asr_line_record[-1])
    #    print("pred text: ", pred_text)
        asr_line_record.append(pred_text)
        asr_line_record.append("")
        asr_wer_lines.append(asr_line_record)
       
    # this is for downstream emsBert and emsBert.tflite
#    transcribed_e2e_lines.insert(0, ["ps_pi_as_si_desc_c_mml_c", "label"])
    file_path = "sampled_liuyi_100e2e_conformerlite_transcribed_test.tsv"
#    file_path = os.path.join("/home/liuyi/tflite_experimental/emsBert/eval_pretrain", "fitted_desc_sampled100e2e_conformerlite_transcribed_test.tsv")
#    write2DListFile(file_path, transcribed_e2e_lines, line_sep = "\t")
    print(file_path, len(transcribed_e2e_lines))
    
    
    asr_wer_lines.insert(0, ["PATH","DURATION","GROUNDTRUTH","GREEDY","BEAMSEARCH"])
    file_path = os.path.join("./sampled100e2e_conformerlite_transcribed.output")
    write2DListFile(file_path, asr_wer_lines, line_sep = "\t")
    print(file_path, len(asr_wer_lines))
    logger.info(f"Evaluating result from {file_path} ...")
    metrics = {
        "greedy_wer": ErrorRate(wer, name="greedy_wer", dtype=tf.float32),
        "greedy_cer": ErrorRate(cer, name="greedy_cer", dtype=tf.float32),
        "beamsearch_wer": ErrorRate(wer, name="beamsearch_wer", dtype=tf.float32),
        "beamsearch_cer": ErrorRate(cer, name="beamsearch_cer", dtype=tf.float32),
    }
    
    asr_wer_lines = asr_wer_lines[1:]
    for eachline in tqdm(asr_wer_lines):
        #_, _, groundtruth, greedy, beamsearch = eachline.split("\t")
        _, _, groundtruth, greedy, beamsearch = eachline
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        greedy = tf.convert_to_tensor([greedy], dtype=tf.string)
        beamsearch = tf.convert_to_tensor([beamsearch], dtype=tf.string)
        metrics["greedy_wer"].update_state(decode=greedy, target=groundtruth)
        metrics["greedy_cer"].update_state(decode=greedy, target=groundtruth)
        metrics["beamsearch_wer"].update_state(decode=beamsearch, target=groundtruth)
        metrics["beamsearch_cer"].update_state(decode=beamsearch, target=groundtruth)
    for key, value in metrics.items():
        print(key, value.result().numpy())
        logger.info(f"{key}: {value.result().numpy()}")
        
if __name__ == "__main__":
    fire.Fire(main)
