# EMS Pipeline
NLP Pipeline for Analysis of EMS Narratives

![GUI](ETC/GUI.png)

This demo under the `Demo/` directory has been succesfully tested on:

`64-bit Ubuntu 16.04 LTS`

`Intel® Core™ i7-7700 CPU @ 3.60GHz × 8`

`Python 2.7` 


### Requirments

Most Python packages could be installed with pip. Some other requirements are:

#### Installing MetaMap:
MetaMap needs to be installed under the `Demo/public_mm` directory. Instruction for installing MetaMap are here:

<https://metamap.nlm.nih.gov/Installation.shtml>

#### Installing PyMetaMap:
PyMetaMap is a Python Wrapper around MetaMap. It needs to be installed in the `Demo/pymetamap` directory. To do this

`cd Demo`

`git clone https://github.com/AnthonyMRios/pymetamap`

`cd pymetamap`

`python setup.py install`

#### Google Cloud Speech API:
To use the Google Cloud Speech API, you need to have your own service account key in JSON format with the Cloud Speech API enabled. It needs to be in the demo folder: 

`Demo/service-account.json`

#### DeepSpeech Models:

DeepSpeech fucntionality is currently * disabled * in the demo. However, the models could be downloaded by:

`mkdir DeepSpeech_Models`

`wget https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz`

`tar xvfz deepspeech-0.5.1-models.tar.gz`

Links to more recent models can be found at: <https://github.com/mozilla/DeepSpeech>

### Running the Demo

Make the `metamap.sh` executable by running:

`chmod +x metamap.sh`

This step is only needed to be run once on every machine.

To run MetaMap, run:

`./metamap.sh`

This step needs to repeated after every reboot.

To run the graphical user interface (GUI), run:

`Python gui.py`

