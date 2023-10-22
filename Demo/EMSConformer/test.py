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
import tensorflow as tf

devices = [0]
gpus = tf.config.list_physical_devices("GPU")
visible_gpus = [gpus[i] for i in devices]
tf.config.set_visible_devices(visible_gpus, "GPU")
# strategy = tf.distribute.MirroredStrategy()



import fire
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import dataset_helpers, exec_helpers, featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")
from datetime import datetime
import argparse

"""
The main function to evaluate the WER and CER of EMSConformer on our EMS audio dataset. It uses EMSConformer to 
transcribe EMS audio dataset and correspondinly compute WER and CER

Input:  1) output: path of the intermediate transcription result output text file
        2) saved: the conformer h5 model used to transcribe the EMS audio files
        3) config: the configuration file to configure the running parameters, including the testset path
Output: the WER and CER for both greedy search and beam search. The beam search is still under development.
"""
def main(
    config: str = DEFAULT_YAML,
    saved: str = None,
    mxp: bool = False,
    bs: int = None,
    sentence_piece: bool = False,
    subwords: bool = True,
    device: int = 0,
    cpu: bool = False,
    output: str = "test.tsv",
):
      
    time_s = datetime.now()

    assert saved and output
    tf.random.set_seed(0)
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": mxp})
#    env_util.setup_devices([device], cpu=cpu)

    config = Config(config)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

#    with strategy.scope():
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer.make(speech_featurizer.shape)
    conformer.load_weights(saved, by_name=True)
    #conformer.summary(line_length=100)
    conformer.add_featurizers(speech_featurizer, text_featurizer)
    
    test_dataset = dataset_helpers.prepare_testing_datasets(
        config=config, speech_featurizer=speech_featurizer, text_featurizer=text_featurizer
    )
    batch_size = bs or config.learning_config.running_config.batch_size
    test_data_loader = test_dataset.create(batch_size)
    
    exec_helpers.run_testing(model=conformer, test_dataset=test_dataset, test_data_loader=test_data_loader, output=output)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

"""
The main function to evaluate the WER and CER of EMSConformer on our EMS audio dataset. It uses EMSConformer to 
transcribe EMS audio dataset and correspondinly compute WER and CER

Input:  1) output: path of the intermediate transcription result output text file
        2) saved: the conformer h5 model used to transcribe the EMS audio files
        3) config: the configuration file to configure the running parameters, including the testset path
Output: the WER and CER for both greedy search and beam search. The beam search is still under development.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "control the functions for conformer")
    #parser.add_argument("--output", action='store', type=str, default = "./test_outputs/librispeech_testclean.tsv", help="the test output for post processing")
    parser.add_argument("--output", action='store', type=str, default = None, help="the test output for post processing")
    parser.add_argument("--saved", action='store', type=str, default = "/slot1/asr_models/tensorflowasr_librispeech_models/tensorflowasr_pretrained/subword-conformer/pretrained-subword-conformer/latest.h5", help="saved model")
    # parser.add_argument("--cuda_device", action='store', type=str, default = "1", help="indicate the cuda device number")
    parser.add_argument("--config", action='store', type=str, default = "config.yml", help="the configuration file for testing")

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    if not args.output:
        test_output_dir = '/'.join(args.saved.split('/')[:-2])
        test_output_path = test_output_dir+'/test_output.tsv'
    else:
        test_output_path = args.output

    print('\n++++test_output_path =', test_output_path)

    #stop

    main(config=args.config, saved=args.saved, output=test_output_path)


