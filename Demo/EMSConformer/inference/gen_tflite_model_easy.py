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

import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#devices = [-1]
#gpus = tf.config.list_physical_devices("GPU")
#visible_gpus = [-1]
#visible_gpus = [gpus[i] for i in devices]
#tf.config.set_visible_devices(visible_gpus, "GPU")
#strategy = tf.distribute.MirroredStrategy()

import fire
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()
#import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import exec_helpers, featurizer_helpers
from tensorflow_asr.models.transducer.conformer import Conformer

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")
from datetime import datetime
import argparse

def main(
    config: str = DEFAULT_YAML,
#    config: str = "./h5_models/subword-conformer-config.yml",
    h5: str = None,
    subwords: bool = True,
    sentence_piece: bool = False,
    output: str = None,
):
    assert h5 and output

    tf.keras.backend.clear_session()
    tf.compat.v1.enable_control_flow_v2()

    config = Config(config)
    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

#    with strategy.scope():
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer.make(speech_featurizer.shape)
    conformer.load_weights(h5, by_name=True)
    #conformer.summary(line_length=100)
    conformer.add_featurizers(speech_featurizer, text_featurizer)

#    print(conformer.inputs)
#    print(conformer.outputs)

    exec_helpers.convert_tflite(model=conformer, output=output)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "control the functions for generating conformer tflite")
    parser.add_argument("--output", action='store', type=str, default=None, help="tflite model output path", required=False)
    parser.add_argument("--h5", action='store', type=str, help="which h5 model to be converted into tflite", required=True)
    parser.add_argument("--config", action='store', type=str, default = "config.yml", help="the configuration file for testing")    

    args = parser.parse_args()

    time_s = datetime.now()

    print()
    print('args.config =', args.config)
    print("args.h5 =", args.h5)

    if not args.output: # infer from the h5 path
        model_dir = "/".join(args.h5.split('/')[:-2]) + '/tflite_model/'
        tflite_model_path = model_dir + 'model.tflite'
    else:
        tflite_model_path = args.output + '/tflite_model/' + 'model.tflite'

    print('tflite_model_path =', tflite_model_path)
    print()
    #stop

    main(config=args.config, h5=args.h5, output=tflite_model_path)

    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)

    #fire.Fire(main)
