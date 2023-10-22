from tensorflow_asr.datasets import asr_dataset
from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import SpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import TextFeaturizer


def prepare_training_datasets(
    config: Config,
    speech_featurizer: SpeechFeaturizer,
    text_featurizer: TextFeaturizer,
    tfrecords: bool = False,
    metadata: str = None,
):
#    print("prepare_training_datasets:")
#    for k, v in vars(config.learning_config.eval_dataset_config).items():
#        print("the value of {} is {}".format(k, v))
#    print(**vars(config.learning_config.eval_dataset_config))
    if tfrecords:
        train_dataset = asr_dataset.ASRTFRecordDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.train_dataset_config),
            indefinite=True
        )
        eval_dataset = asr_dataset.ASRTFRecordDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.eval_dataset_config),
            indefinite=True
        )
    else:
        train_dataset = asr_dataset.ASRSliceDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.train_dataset_config),
            indefinite=True
        )
        eval_dataset = asr_dataset.ASRSliceDataset(
            speech_featurizer=speech_featurizer,
            text_featurizer=text_featurizer,
            **vars(config.learning_config.eval_dataset_config),
            indefinite=True
        )
    train_dataset.load_metadata(metadata)
    eval_dataset.load_metadata(metadata)
    return train_dataset, eval_dataset


def prepare_testing_datasets(
    config: Config,
    speech_featurizer: SpeechFeaturizer,
    text_featurizer: TextFeaturizer,
):
    test_dataset = asr_dataset.ASRSliceDataset(
        speech_featurizer=speech_featurizer,
        text_featurizer=text_featurizer,
        **vars(config.learning_config.test_dataset_config)
    )
    return test_dataset


def prepare_training_data_loaders(
    config: Config,
    train_dataset: asr_dataset.ASRDataset,
    eval_dataset: asr_dataset.ASRDataset,
    batch_size: int = None,
):
    global_batch_size = batch_size or config.learning_config.running_config.batch_size
    train_data_loader = train_dataset.create(global_batch_size)
    eval_data_loader = eval_dataset.create(global_batch_size)
#    print("train_data_loader: ", train_data_loader.cardinality().numpy())
#    print("eval_data_loader: ", eval_data_loader.cardinality().numpy())
    return train_data_loader, eval_data_loader, global_batch_size
