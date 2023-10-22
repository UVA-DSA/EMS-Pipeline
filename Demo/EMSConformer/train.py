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
devices = [0]
gpus = tf.config.list_physical_devices("GPU")
visible_gpus = [gpus[i] for i in devices]
tf.config.set_visible_devices(visible_gpus, "GPU")
#strategy = tf.distribute.MirroredStrategy()

import os
import fire
import math
from tensorflow_asr.utils import env_util

logger = env_util.setup_environment()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.helpers import featurizer_helpers, dataset_helpers
from tensorflow_asr.models.transducer.conformer import Conformer
from tensorflow_asr.optimizers.schedules import TransformerSchedule


DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")
from datetime import datetime
import argparse

def main(
    config: str = DEFAULT_YAML,
    tfrecords: bool = False,
    sentence_piece: bool = False,
    subwords: bool = True,
    bs: int = None,
    spx: int = 1,
    metadata: str = None,
    static_length: bool = False,
#    devices: list = [2],
    mxp: bool = True,
    pretrained: str = None,
):

      
    time_s = datetime.now()

    tf.keras.backend.clear_session()
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": mxp})
#    print(devices)

    config = Config(config)

    speech_featurizer, text_featurizer = featurizer_helpers.prepare_featurizers(
        config=config,
        subwords=subwords,
        sentence_piece=sentence_piece,
    )

    train_dataset, eval_dataset = dataset_helpers.prepare_training_datasets(
        config=config,
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        tfrecords=tfrecords,
        metadata=metadata,
    )

    if not static_length:
        speech_featurizer.reset_length()
        text_featurizer.reset_length()
#        print("speech_featurizer max length = %s, text_featurizer max_length = %s" % (speech_featurizer.max_length, text_featurizer.max_length))

    train_data_loader, eval_data_loader, global_batch_size = dataset_helpers.prepare_training_data_loaders(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=bs,
    )


#    with strategy.scope():
    #    print("conformer output vocab size: %s" % text_featurizer.num_classes)
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer.make(speech_featurizer.shape, prediction_shape=text_featurizer.prepand_shape, batch_size=global_batch_size)
    if pretrained:
        conformer.load_weights(pretrained, by_name=True, skip_mismatch=True)
    conformer.summary(line_length=100)
    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=conformer.dmodel,
            warmup_steps=config.learning_config.optimizer_config.pop("warmup_steps", 10000),
            max_lr=(0.05 / math.sqrt(conformer.dmodel)),
        ),
        **config.learning_config.optimizer_config
    )
    conformer.compile(
        optimizer=optimizer,
        experimental_steps_per_execution=spx,
        global_batch_size=global_batch_size,
        blank=text_featurizer.blank,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(**config.learning_config.running_config.checkpoint),
        tf.keras.callbacks.EarlyStopping(patience=20, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.experimental.BackupAndRestore(config.learning_config.running_config.states_dir),
        tf.keras.callbacks.TensorBoard(**config.learning_config.running_config.tensorboard),
    ]

    conformer.fit(
        train_data_loader,
        epochs=config.learning_config.running_config.num_epochs,
        validation_data=eval_data_loader,
        callbacks=callbacks,
        steps_per_epoch=train_dataset.total_steps,
        validation_steps=eval_dataset.total_steps if eval_data_loader else None,
    )


    time_t = datetime.now() - time_s
    print("This run takes %s" % time_t)    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "control the functions for conformer")
#    parser.add_argument("--pretrained", action='store', type=str, default = "/slot1/asr_models/tensorflowasr_librispeech_models/tensorflowasr_pretrained/subword-conformer/pretrained-subword-conformer/latest.h5", help="pretrained model")
    parser.add_argument("--pretrained", action='store', type=str, default = None, help="pretrained model")
    parser.add_argument("--config", action='store', type=str, default = "config.yml", help="the configuration file for testing")

    args = parser.parse_args()

    main(config=args.config, pretrained=args.pretrained)

