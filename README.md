# CognitiveEMS Pipeline

## About This Project
This repository contains the decision support pipeline for a real-time, multimodal, and edge-deployed Cognitive Assistant System for Emergency Medical Serivces (EMS). The system aims to improve situational awareness of the first responders/paramedics by automated collection and analysis of multimodal data from incident scenes and providing suggestions to them. The figure below shows the high level overview of the proposed system. For more information visit the [project page](https://www.nist.gov/ctl/pscr/cognitive-assistant-systems-emergency-response).

![Architecture](<README Images/CognitiveEMS.png>)

## Notice
:exclamation: The branches related previous work carried out in this project is archived and you may browse them under tags.

## Branches in This Repository 
- `master` : the CognitiveEMS pipeline implemented on server (Ubuntu desktop). used to evaluate the pipeline performance and generate results displayed in our paper.
- `edge` : the CognitiveEMS pipeline deployed on edge (Jetson Nano) used to evaluate the pipeline performance and generate results displayed in our paper.
- `demo` : a portable demo of the CognitiveEMS pipeline with a GUI implemented for an Ubuntu based system that can be used for demonstration of the Cognitive Assistant.
- `dcs` : a data collection system developed to collect multimodal data.
    - supported modalities: Video, Audio, IMU Data from smartwatch
    - supported devices: 
        - Android smartphone/smarglass for Video and Audio
        - GoPro for Video and Audio
        - WearOS based smartwatches for IMU data

## Publically Available Datasets
One of the challenges of developing a Cognitive Assistant system for the EMS domain is a lack of realistic audio and video EMS data. We contribute new publicly-available datasets, including human and synthetic EMS conversational audio and multimodal data from simulated EMS scenarios that can be found [here]().

## Models and Software Architecture  
The figure below illustrates a high level overview of the software architecture. The pipeline is a real-time multi-thread system. There is a main process that handles execution and coordination of the three AI models running on their own separate threads: (1) EMS-Whisper for Speech Recognition, which transcribes audio input into a text transcript; (2) EMS-TinyBERT for Protocol Prediction, which predicts the correct protocol for the incident from the text transcript; and (3) EMS-Vision for Intervention Recognition, which recognizes the medical treatment action based on video input and the predicted protocol. Each model is described in more detail below. Further architecture details on the folder structure and code can be found in the [Technical Documentation](#technical-documentation) section. 

![Software Design](<README Images/Overall Figure.png>)

### EMS-Whisper for Speech Recognition
Our speech recognition model, EMS-Whisper, is a fine-tuned version of [Whisper](https://openai.com/research/whisper) models trained on a curated dataset of conversational EMS speech. We have two sizes of EMS-Whisper: base-finetuned (39M parameters) and tiny-finetuned (74M parameters), which are the finetuned English-only tiny and base Whisper models. We choose to build off the Whisper architecture because it has been found to be robust to noisy environments, diverse ranges of accents, and has strong out-of-the-box performance on numerous datasets. A separate github repository with more details about EMS-Whisper can be found [here](https://github.com/UVA-DSA/EMS-Whisper).


### EMS-TinyBERT for Protocol Prediction
Our protocol prediction model, EMSTinyBERT, consists of three parts: (1) [TinyClinicalBERT](https://huggingface.co/nlpie/tiny-clinicalbert), a [BERT](https://arxiv.org/abs/1810.04805) model pretrained on medical corpus to encoder to encode EMS transcripts into text features; (2) a graph neural network to fuse domain knowledge with text features; and (3) a group-wise training strategy to deal with scarcity of training samples in rare classes. EMS-TinyBERT is a lightweight model that can be deployed on edge. A separate github repository with more details about EMS-TinyBERT can be found [here](https://github.com/UVA-DSA/DKEC).

### EMS-Vision for Intervention Recognition

Our intervention recognition model, EMS-Vision, consists
of three parts: (1) The protocol prediction from EMS-
TinyBERT give contextual basis to the (2) knowledge agent that provides a subset of interventions associated with the protocol to (3) [CLIP](https://openai.com/research/clip), an open-source zero-shot image classification model. A separate github repository with more details about EMS-Vision can be found [here]().


### Additional Options for Models

#### Google Cloud Speech-To-Text API for Speech Recognition
There is an option to use the [Google Cloud Speech-To-Text API](https://cloud.google.com/speech-to-text/) for speech recognition in the CognitiveEMS pipeline. However, this option relies on a cloud service and the pipeline will no longer be edge-deployed.  

#### EMSAssist Models
There is another Cognitive Assistant pipeline for EMS called [EMSAssist](https://dl.acm.org/doi/abs/10.1145/3581791.3596853). We have integrated EMSAssist's Speech Recognition Model (EMSConformer) and Protocol Prediction Model (EMSMobileBERT) for evaluation purposes to compare our pipeline to the current state of the art. EMSAssist's github repositories can be found [here](https://github.com/LENSS/EMSAssist) and [here](https://github.com/liuyibox/EMSAssist-artifact-evaluation).

## Getting Started 
### Prerequisites
To get started CognitiveEMS, you need to have [`Conda`](https://docs.conda.io/en/latest/) and [`git`](https://git-scm.com/) installed. All our implementions used **Conda version 23.1.0** and **git version 2.25.1**. We list hardware specifications for each device we used below. You may be able to use different hardware specifications and different versions of CUDA.

The `server` implementation hs been successfully tested on an Ubuntu desktop with these specifications:
- OS: **64-bit Ubuntu 20.04.6 LTS**
- Processor: **13th Gen Intel® Core™ i9-13900KF x 32**
- Graphics: **NVIDA Corporation**
- NVIDIA Cuda complier driver: **version 10.1.243**
- CUDA: **version 12.1**

The `edge` implementation hs been successfully tested on a Jetson Nano with these specifications:
- Jetson nano

The `demo` implementation hs been successfully tested on an Alienware m15 R7 laptop with these specifications:
- OS: **64-bit Ubuntu 20.02.6 LTS**
- Processor: **12th Gen Intel® Core™ i7-12700H x 20**
- Graphics: **NVIDIA Corporation / Mesa Intel® Graphics (ADL GT2)**
- NVIDIA Cuda complier driver: **version 11.3.58**
- CUDA: **version 11.4**

You may be able to get CognitiveEMS running on hardware with different specifications and different verisions of CUDA.

### Installation
Clone our repository from github and go to the server branch
```bash
# clone repo
git clone https://github.com/UVA-DSA/CognitiveEMS
cd CognitiveEMS/

# go to server branch
git checkout server 
```

### Conda Environment Setup 
Create and activate the Conda environment that uses `Python 3.8.18 `named `CogEMS` for dependencies and packages.

If you are using hardware that matches **[our specifications](#prerequisites) (Linux OS, CUDA 12.1),** create the CogEMS Conda environment from the `environment.yml` file.
```bash
# create Conda environment from yaml file
cd Pipeline/
conda env create --file environment.yml

# activate Conda environment
conda activate CogEMS
```

If you are using hardware with **<span style="color:red">different specifications</span>,** you may have to manually create the CogEMS Conda environment.
```bash
# manually create conda enviroment
conda create --name CogEMS python=3.8.18

# activate Conda environment
conda activate CogEMS

# Now, manually install dependencies
# check cuda version
nvidia-smi

# execute pytorch install command
'''
go to https://pytorch.org/get-started/locally/ and get the installation command for these options:
Stable (2.1.0), Linux, conda, Python, and your CUDA version 
'''
# for the our server, the command was:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# install other packages
conda install pyaudio
pip install torch-geometric
pip install pyyaml
pip install transformers
pip install chardet
pip install charset-normalizer==3.1.0
pip install pyqt5
pip install pandas
pip install openpyxl
pip install evaluate
pip install jiwer
```
### Virtual Speaker and Mic Setup
Since the CognitiveEMS pipeline is real-time and takes in a stream of audio from a mic as input, we set up a virtual speaker and mic to capture the input internally for our audio recordings.

> **<span style="color:red">NOTE!</span>** You have to run the commands to set up the virtual speaker and mic every time you restart your device.

```bash
# set up virtual speaker
pactl load-module module-null-sink sink_name="virtual_speaker" sink_properties=device.description="virtual_speaker"

# set up virtual mic
pactl load-module module-remap-source master="virtual_speaker.monitor" source_name="virtual_mic" source_properties=device.description="virtual_mic"
```

### Model Setup
Since model weight files were too large to store on our github repository, you need to download them from our publically available Box folders for EMS-Whisper and EMS-TinyBERT.

#### Setup for EMS-Whisper
Download the `models` folder for EMS-Whisper from [here]() and put the `models` folder under the `Pipeline/EMS_Whisper` folder as: `Pipeline/EMS_Whisper/models/`

#### Setup for EMS-TinyBERT
Download the `models` folder for EMS-TinyBERT from [here]() and put the `models` folder under the `Pipeline/EMS_TinyBERT` folder as: `Pipeline/EMS_TinyBERT/models/`

### Google Cloud Speech-To-Text API Setup (Optional)
To use the Google Cloud Speech-To-Text API, you need to have your own service account key in a JSON file. The service account must have the Speech API enabled. Put your JSON file under the Pipeline folder as: `Pipeline/service-account.json`. For more information and to get started with Google Cloud Speech-To-Text go [here](https://cloud.google.com/speech-to-text/).

## Usage
### Running the Pipeline

#### Pipeline Options

### Running Evaluations of the Pipeline
#### Reproducing Results


## Technical Documentation
Coming soon

## Video Demonstrations


https://github.com/UVA-DSA/EMS-Pipeline/assets/40396880/8511b697-c680-4af9-9e1c-8681b7d7cdc9




## Publications
[“Real-Time Multimodal Cognitive Assistant for Emergency Medical Services”](https://arxiv.org/pdf/2403.06734.pdf)  
K. Weerasinghe, S. Janapati, X. Ge, S. Kim, S. Iyer, J. Stankovic, H. Alemzadeh  
In the 9th ACM/IEEE Conference on Internet of Things Design and Implementation (IoTDI), 2024.  

["DKEC: Domain Knowledge Enhanced Multi-Label Classification for Electronic Health Records"](https://arxiv.org/pdf/2310.07059.pdf)  
X. Ge, R. D. Williams, J. A. Stankovic, H. Alemzadeh  
Preprint available on arXiv, 2024.   

["Camera-Independent Single Image Depth Estimation From Defocus Blur"](https://openaccess.thecvf.com/content/WACV2024/papers/Wijayasingha_Camera-Independent_Single_Image_Depth_Estimation_From_Defocus_Blur_WACV_2024_paper.pdf)  
L. Wijayasingha, H. Alemzadeh, J. A. Stankovic  
In the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024.  

[“emsReACT: A Real-Time Interactive Cognitive Assistant for Cardiac Arrest Training in Emergency Medical Services”](https://homa-alem.github.io/papers/DCOSS_IOT_2023_emsReACT.pdf)  
M A. Rahman, L. Jia, E. Mirza, S. M. Preum. H. Alemzadeh, R. D. Williams, J. Stankovic  
In the 19th Annual Int. Conf. on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT), 2023.  

["Poster Abstract: SenseEMS - Towards A Hand Activity Recognition and Monitoring System for Emergency Medical Services](https://homa-alem.github.io/papers/IPSN_Poster_2023_SenseEMS.pdf)    
M. A. Rahman, K. Weerasinghe, L. Wijayasingha, H. Alemzadeh, R. D. Williams, J. Stankovic  
In the 22nd annual ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN), 2023.  

[“EMS-BERT: A Pre-Trained Language Representation Model for the Emergency Medical Services (EMS) Domain”](https://homa-alem.github.io/papers/CHASE_2023_EMS-BERT.pdf)  
M. A. Rahman, S. M. Preum, R. D. Williams, H. Alemzadeh, J. Stankovic  
In the IEEE/ACM Conf. on Connected Health: Applications, Systems and Engineering Technologies (CHASE), 2023.  

[“Information Extraction from Patient Care Reports for Intelligent Emergency Medical Services”](https://homa-alem.github.io/papers/CHASE_2021.pdf)  
S. Kim, W. Guo, R. Williams, J. Stankovic, H. Alemzadeh  
In the IEEE/ACM Conf. on Connected Health: Applications, Systems, and Engineering Technologies (CHASE), 2021. **(Best Paper Finalist)** 

[“A Review of Cognitive Assistants for Healthcare: Trends, Prospects, and Future Directions”](https://www.cs.virginia.edu/~stankovic/psfiles/CognitiveAssistantHealthSurvey_Main.pdf)  
S. Preum, S. Munir, M. Ma, M. S. Yasar, D. J. Stone, R. Williams, H. Alemzadeh, J. Stankovic  
In ACM Computing Surveys, 2021.  

[“IMACS-an interactive cognitive assistant module for cardiac arrest cases in emergency medical service: Demo Abstract.”](https://dl.acm.org/doi/abs/10.1145/3384419.3430451)  
M. A. Rahman, S. Preum, J. Stankovic. L. Jia, E. Mirza, R. Williams, H. Alemzadeh  
In the 18th Conference on Embedded Networked Sensor Systems (SenSys'20), 2020.

[“EMSContExt: EMS Protocol-driven Concept Extraction for Cognitive Assistance in Emergency Response”](https://homa-alem.github.io/papers/EMSContExt_IAAI2020.pdf)  
S. Preum, S. Shu, H. Alemzadeh, J. Stankovic  
In the Thirty-Second Annual Conf. on Innovative Applications of Artificial Intelligence (IAAI-20), 2020.

[“GRACE: Generating Summary Reports Automatically for Cognitive Assistance in Emergency Response”](https://www.cs.virginia.edu/~stankovic/psfiles/IAAI-RahmanM.42.pdf)  
M. A. Rahman, S. M. Preum, R. Williams, H. Alemzadeh, J. Stankovic  
In the Thirty-Second Annual Conf. on Innovative Applications of Artificial Intelligence (IAAI-20), 2020.

["A Behavior Tree Cognitive Assistant System for Emergency Medical Services](https://homa-alem.github.io/papers/IROS2019.pdf)  
S. Shu, S. Preum, H. M. Pitchford, R. D. Williams, J. Stankovic, H. Alemzadeh  
In the IEEE IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019

["CognitiveEMS: A Cognitive Assistant System for Emergency Medical Services"](https://homa-alem.github.io/papers/MEDCPS_2018.pdf)  
S. Preum, S. Shu, M. Hotaki, R. Williams, J. Stankovic, H. Alemzadeh  
In SIGBED Review, Special Issue on Medical Cyber Physical Systems Workshop (CPS-Week), 2018.  
**Featured by the IWCE's Urgent Communications and UVA SEAS News, 2018.**
 
["Towards a Cognitive Assistant System for Emergency Response"](https://homa-alem.github.io/papers/ICCPS_Poster_2018.pdf)  
S. Preum, S. Shu, J. Ting, V. Lin, R. Williams, J. Stankovic, H. Alemzadeh  
In the 9th ACM/IEEE International Conference on Cyber-Physical Systems (CPS-Week), 2018.

