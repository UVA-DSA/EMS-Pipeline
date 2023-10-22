from datetime import datetime
import argparse

import os
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

import tensorflow as tf
#from tensorflow_asr.utils import app_util
#from tensorflow_asr.helpers import exec_helpers
from tqdm import tqdm
import numpy as np

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
    
    #logger = tf.get_logger()
    logger.info(f"Evaluating result from {filepath} ...")
    metrics = {
        "wer": ErrorRate(wer, name="wer", dtype=tf.float32),
        "cer": ErrorRate(cer, name="cer", dtype=tf.float32),
    }
    with read_file(filepath) as path:
        with open(path, "r", encoding="utf-8") as openfile:
            lines = openfile.read().splitlines()
   
    # metrics["wer"].reset_state()
    # metrics["cer"].reset_state()

    for eachline in tqdm(lines):
        # print("eachline =", eachline)
        # print("len(eachline) =", len(eachline.split("\t")))
        groundtruth, asr_output = eachline.split("\t")
        groundtruth = tf.convert_to_tensor([groundtruth], dtype=tf.string)
        asr_output = tf.convert_to_tensor([asr_output], dtype=tf.string)
        metrics["wer"].update_state(decode=asr_output, target=groundtruth)
        metrics["cer"].update_state(decode=asr_output, target=groundtruth)
    for key, value in metrics.items():
        logger.info(f"{key}: {value.result().numpy()}")

    res_wer = metrics["wer"].result().numpy()
    res_cer = metrics["cer"].result().numpy()

    return res_wer, res_cer
    

fn_list = [
    'eval_GC_transcript_default.txt',
    'eval_GC_transcript_latest_long.txt',
    'eval_GC_transcript_latest_short.txt',
    'eval_GC_transcript_video.txt',
    'eval_GC_transcript_phone_call.txt',
    'eval_GC_transcript_command_and_search.txt',
    'eval_GC_transcript_medical_dictation.txt',
    'eval_GC_transcript_medical_conversation.txt',    
]

spk_dirs = [
    "sample_tian2",
    "sample_liuyi",
    "sample_yichen",
    "sample_radu",
    "sample_amran",
    "sample_michael"
]   

# python evaluate_asr_tian.py --dir 

"""
The main function to evaluate the WER and CER of transcription from Google Cloud Speech-to-Text. 
It's good to note here we directly use the transcribed texts from Google Cloud Speech-to-Text API for evaluation.

Input: the directory contains the transcribed texts from 8 Google Cloud Speech-to-Text APIs on all speaker voice
Output: the averaged WER and CER
"""
if __name__ == "__main__":
      
    time_s = datetime.now()

    parser = argparse.ArgumentParser(description = "control the functions for conformer")
    parser.add_argument("--dir", action='store', type=str, help="Directory containing the eval_*.txt files for different models", required=False)
    parser.add_argument("--result_file", action='store', type=str, help="get the wer and cer", required=False)

    args = parser.parse_args()

    if args.result_file:
        # compare_test_output(args.result_file)
        compare_normal_output(args.result_file)
    
    wer_lists = []
    cer_lists = []
    if args.dir:

        for spk_dir in spk_dirs:
            spk_dir_path = os.path.join(args.dir, spk_dir)

            wer_list = []
            cer_list = []
            for filename in fn_list:
                print(filename)
                res_wer, res_cer = compare_normal_output(os.path.join(spk_dir_path, filename))
                wer_list.append(res_wer)
                cer_list.append(res_cer)
                # print('WER =', res_wer)
                # print('CER =', res_cer)
                # print()
            wer_lists.append(wer_list)
            cer_lists.append(cer_list)
            # write fn_list, wer_list, cer_list for 3 columns to a csv file
            # import pandas as pd
            # df = pd.DataFrame({'filename': fn_list, 'wer': wer_list, 'cer': cer_list})
            # df.to_csv(os.path.join(args.dir, 'eval_wer_cer.csv'), index=False)

    # print(wer_lists)
    # print(cer_lists)

    averaged_wers = np.average(wer_lists, axis = 0)
    # print(averaged_wers)
    averaged_cers = np.average(cer_lists, axis = 0)
    # print(averaged_cers)

    for idx, wer in enumerate(averaged_wers):
        cer = averaged_cers[idx]
        print("GC%d\t\t%s\t\t%s" % (idx + 1, wer, cer))

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)
