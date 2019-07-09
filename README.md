# EMS Pipeline
Real-Time NLP Pipeline for Analysis of EMS Narratives

## Pipeline Demo graphical User Interface

![GUI](ETC/GUI.png)

This demo under the `Demo/` directory has been succesfully tested on:

`64-bit Ubuntu 16.04 LTS`

`Intel® Core™ i7-7700 CPU @ 3.60GHz × 8`

`Python 2.7` 

### Requirements

Most Python packages could be installed with pip. Some other requirements are:

#### Installing MetaMap:
MetaMap needs to be installed under the `Demo/public_mm` directory. Downloads are hosted at:

<https://metamap.nlm.nih.gov/MainDownload.shtml>

You will need a UMLS account/license. You can request one here: 

<https://uts.nlm.nih.gov/license.html>

Instruction for installing MetaMap are here: 

<https://metamap.nlm.nih.gov/Installation.shtml>

#### Installing PyMetaMap:
PyMetaMap is a Python Wrapper around MetaMap. It needs to be installed in the `Demo/pymetamap` directory. The software is already in this directory, but needs to be built:

`cd pymetamap`

`python setup.py install`

For more information, visit: <https://github.com/AnthonyMRios/pymetamap>

#### Google Cloud Speech API:
To use the Google Cloud Speech API, you need to have your own service account key in JSON format. The service account must have the Speech API enabled enabled. It needs to be in the demo folder: 

`Demo/service-account.json`

#### DeepSpeech Models:

DeepSpeech fucntionality is currently * disabled * in the demo. The models are no need, but, they could be downloaded:

`mkdir DeepSpeech_Models`

`wget https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz`

`tar xvfz deepspeech-0.5.1-models.tar.gz`

For more information and link to more recent models, visit: <https://github.com/mozilla/DeepSpeech>

### Running the Demo

Make the `metamap.sh` executable by running (This step is only needed to be run once on every machine):

`chmod +x metamap.sh`

To run MetaMap, run (This step needs to repeated after every reboot):

`./metamap.sh`

To launch the graphical user interface (GUI), run:

`Python GUI.py`

## Publications

["CognitiveEMS: A Cognitive Assistant System for Emergency Medical Services"](http://faculty.virginia.edu/alemzadeh/papers/MEDCPS_2018.pdf)  
S. Preum, S. Shu, M. Hotaki, R. Williams, J. Stankovic, H. Alemzadeh
In SIGBED Review, Special Issue on Medical Cyber Physical Systems Workshop (CPS-Week), 2018.
 Featured by the IWCE's Urgent Communications and UVA SEAS News, 2018.
 
["Towards a Cognitive Assistant System for Emergency Response"](http://faculty.virginia.edu/alemzadeh/papers/ICCPS_Poster_2018.pdf)  
S. Preum, S. Shu, J. Ting, V. Lin, R. Williams, J. Stankovic, H. Alemzadeh
In the 9th ACM/IEEE International Conference on Cyber-Physical Systems (CPS-Week), 2018.
