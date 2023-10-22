from datetime import datetime
import argparse

import os
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

import tensorflow as tf
#from tensorflow_asr.utils import app_util
#from tensorflow_asr.helpers import exec_helpers
from tqdm import tqdm

devices = [0]
gpus = tf.config.list_physical_devices("GPU")
visible_gpus = [gpus[i] for i in devices]
tf.config.set_visible_devices(visible_gpus, "GPU")

from tensorflow_asr.metrics.error_rates import ErrorRate
from tensorflow_asr.utils.file_util import read_file
from tensorflow_asr.utils.metric_util import cer, wer

logger = tf.get_logger()

"""
compare_test_output is used to evaluate wer and cer given a text file from tensorflow_asr code. The tensorflow_asr
uses both greedy search and beam search to generate transcribed texts. The greedy search is working, and the beam 
search is under development.

Input: the path of text file with transcribed text and ground true text for both greedy search and beam search.
Output: the WER and CER of the given text file for both greedy search and beam search.
"""
def compare_test_output(filepath: str):

    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "greedy_wer": ErrorRate(wer, name="greedy_wer", dtype=tf.float32),
        "greedy_cer": ErrorRate(cer, name="greedy_cer", dtype=tf.float32),
        "beamsearch_wer": ErrorRate(wer, name="beamsearch_wer", dtype=tf.float32),
        "beamsearch_cer": ErrorRate(cer, name="beamsearch_cer", dtype=tf.float32),
    }
    with read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
            lines = lines[1:]  # skip header
#    clean_lines = []
    for eachline in tqdm(lines):
        _, _, groundtruth, greedy, beamsearch = eachline.split("\t")
#        clean_lines.append(groundtruth + "\t" + greedy)
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        greedy = tf.convert_to_tensor([greedy], dtype=tf.string)
        beamsearch = tf.convert_to_tensor([beamsearch], dtype=tf.string)
        metrics["greedy_wer"].update_state(decode=greedy, target=groundtruth)
        metrics["greedy_cer"].update_state(decode=greedy, target=groundtruth)
        metrics["beamsearch_wer"].update_state(decode=beamsearch, target=groundtruth)
        metrics["beamsearch_cer"].update_state(decode=beamsearch, target=groundtruth)
    for key, value in metrics.items():
        logger.info(f"{key}: {value.result().numpy()}")

"""
compare_normal_output is used to evaluate wer and cer given a normal text file 
with only transcribed text and ground true text.

Input: the path of text file with only transcribed text and ground true text.
Output: the WER and CER for the given text file
"""
def compare_normal_output(filepath: str):

    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "wer": ErrorRate(wer, name="wer", dtype=tf.float32),
        "cer": ErrorRate(cer, name="cer", dtype=tf.float32),
    }
    with read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
    for eachline in tqdm(lines):
        groundtruth, asr_output = eachline.split("\t")
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        asr_output = tf.convert_to_tensor([asr_output], dtype=tf.string)
        metrics["wer"].update_state(decode=asr_output, target=groundtruth)
        metrics["cer"].update_state(decode=asr_output, target=groundtruth)
    for key, value in metrics.items():
        logger.info(f"{key}: {value.result().numpy()}")
    

"""
The main function to evaluate WER and CER of transcription from tensorflow_asr models, including EMSConformer,
ContextNet, rnn_transducer.  

Input: the path of the transcription result from EMSConformer
Output: the WER and CER
"""
if __name__ == "__main__":
      
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for conformer")
    parser.add_argument("--result_file", action='store', type=str, default = "/home/liuyi/TensorFlowASR/examples/conformer/test_outputs/librispeech_testems.tsv", help="get the wer and cer", required=True)

    args = parser.parse_args()

    # compare_test_output(args.result_file)
    compare_normal_output(args.result_file)
    
    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

