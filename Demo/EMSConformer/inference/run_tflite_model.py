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

import fire
import tensorflow as tf
import json
import os
from pydub import AudioSegment

from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio


def main(
#    filename: str = "/home/liuyi/TensorFlowASR/dataset/LibriSpeech/test-clean/5639/40744/5639-40744-0008.flac",
    filename: str = "/home/liuyi/audio_data/sample100/signs_symptoms_audio_concatenated/sss81.wav",
    tflite: str = "./tflite_models/pretrained_librispeech_train_ss_test_concatenated_epoch50_noOptimize.tflite",
#    tflite: str = "./tflite_models/pretrained_librispeech_train_ss_test_concatenated_epoch50.tflite",
#    tflite: str = "./tflite_models/pretrained_librispeech_train_ss_test_concatenated_epoch50_noSELECT_TF_OPS.tflite",
#    tflite: str = "./tflite_models/subsampling-conformer.latest.tflite",
#    tflite: str = "",
    blank: int = 0,
    num_rnns: int = 1,
    nstates: int = 2,
    statesize: int = 320,
):
    tflitemodel = tf.lite.Interpreter(model_path=tflite)

    if os.path.splitext(filename)[1] == ".m4a":
        track = AudioSegment.from_file(filename, "m4a")
        wav_f_path = filename.replace("m4a", "wav")
        wav_file_handle = track.export(wav_f_path, format='wav')
        filename = wav_f_path

    signal = read_raw_audio(filename)
#    print(type(signal))
#    print(signal.shape)

    input_details = tflitemodel.get_input_details()
    print("input details:\n")
    print(input_details)
    print("resized input sh")
#    input_json_obj = json.loads(input_details[0])
#    input_json_formatted_str = json.dumps(json_obj, indent = 2)
#    print(input_json_formmated_str)

    output_details = tflitemodel.get_output_details()
    print("\noutput details:\n")
    print(output_details)
    tflitemodel.resize_tensor_input(input_details[0]["index"], signal.shape)
    tflitemodel.allocate_tensors()
    tflitemodel.set_tensor(input_details[0]["index"], signal)
    tflitemodel.set_tensor(input_details[1]["index"], tf.constant(blank, dtype=tf.int32))
    tflitemodel.set_tensor(input_details[2]["index"], tf.zeros([num_rnns, nstates, 1, statesize], dtype=tf.float32))
    tflitemodel.invoke()
    hyp0 = tflitemodel.get_tensor(output_details[0]["index"])
    hyp1 = tflitemodel.get_tensor(output_details[1]["index"])
    hyp2 = tflitemodel.get_tensor(output_details[2]["index"])

    print("output:\n")
#    print(hyp0)
#    print(hyp1)
#    print(hyp2)
    print("".join([chr(u) for u in hyp0]))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    fire.Fire(main)
