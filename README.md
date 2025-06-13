# CognitiveEMS Pipeline

## About This Project

![CognitiveEMS](Assets/CognitiveEMS_Arch.png)


The **CognitiveEMS Pipeline** is a decision support system designed for real-time, multimodal, and edge-deployed **Cognitive Assistant Systems for Emergency Medical Services (EMS)**. Its primary goal is to enhance situational awareness for first responders and paramedics by automatically collecting and analyzing multimodal data from incident scenes, providing actionable insights and suggestions. The system comprises advanced AI models and modular software components designed for deployment on edge devices.

For more information, visit the [project page](https://www.nist.gov/ctl/pscr/cognitive-assistant-systems-emergency-response).

---

## Demo Branch Overview

This branch contains a **portable demo** of the CognitiveEMS pipeline, featuring a GUI for Ubuntu-based systems. The demo includes:

### Server Features
- Real-time **speech recognition**.
- **Protocol prediction** based on incident data.
- **EMS object detection**.
- Preliminary version of **EMS intervention detection**.

### AR Smartglass Features
- **Augmented Reality (AR)** feedback displaying:
  - Predicted protocols.
  - Detected EMS objects.
  - Detected interventions.
- Real-time streaming of egocentric video and audio.

### Smartwatch Features
- Real-time streaming of **hand movement data**.

---

## High-Level Architecture

The CognitiveEMS pipeline employs a real-time, multi-threaded architecture to process multimodal data inputs. Its main components include:

### Core AI Models
1. **EMS-Whisper**: A fine-tuned speech recognition model for EMS audio inputs.
2. **EMS-TinyBERT**: A protocol prediction model utilizing medical knowledge and transcript analysis.
3. **EMS-Vision**: An intervention recognition model leveraging contextual knowledge and video data.


For detailed descriptions of the models, refer to the [Technical Documentation](#technical-documentation).

---

## Hardware Requirements

### Recommended Specifications

#### Server
- **Processor**: 12th Gen (or newer) Intel® Core™ i5, i7, or i9.
- **RAM**: 32GB.
- **Graphics**: NVIDIA RTX 3080 or higher.

#### Smartglass
- **Model**: [Vuzix M4000 Smart Glasses](https://www.vuzix.com/products/m4000-smart-glasses).

#### Smartwatch
- **Model**: Samsung Galaxy Watch 5.

---

### Minimum Specifications

#### Server
- **Processor**: 12th Gen Intel® Core™ i5
- **RAM**: 16GB.
- **Graphics**: NVIDIA RTX 3060

---

## Software Requirements

#### Server
- **OS**: Ubuntu 22.04.5 LTS (64-bit).
- **CUDA**: Version 12.4.
- **Environment**: Conda (Python 3.10.9).
- **Android Studio**: For android app deployments for smartglass, smartwatch.

#### Smartglass
- **OS**: Android OS.

#### Smartwatch
- **OS**: Android Wear OS.

---

## Prerequisites

### Network Configuration
- Ensure the server, smartglass, and smartwatch are on the **same network**.
- Assign a **static IP** to the server and configure it in the Android applications.
- Remove any firewall rules blocking TCP or UDP traffic.

![Network Architecture](Assets/Network_Architecture.png)

---

## Installation 

:exclamation: Following instructions are still a work in progress.



### 1. Clone the Repository
```bash
git clone https://github.com/UVA-DSA/EMS-Pipeline.git
git checkout demo
```

### 2. Conda Environment Setup
#### Standard Setup
For hardware matching [recommended specifications](#hardware-requirements), use the provided `environment.yml` file:
```bash
cd Demo/

# Make sure your system is up-to-date
sudo apt update
sudo apt upgrade

# dependencies for pyaudio
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools

```

#### Custom Setup

Manually install dependencies:
1. Identify your CUDA version using `nvidia-smi`.
2. Use the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch (v2.5.1 tested), torchvision, and torchaudio for your system.
```bash

conda create -n EMSProject python=3.10
conda activate EMSProject
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# downgrade below packages in the given order.
pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install -q torch-geometric==2.3.1
```

3. Install additional packages:
```bash
pip install pyyaml transformers==4.27.2 pyqt5 pandas openpyxl evaluate jiwer
sudo apt-get install libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev

pip install pyaudio
pip install nltk
pip install python-socketio
pip install google-cloud-speech
pip install pygame
pip install sounddevice
pip install mediapipe
pip install netifaces
pip install opencv-python-headless
pip install pyqt5==5.15.6
pip install py-trees==2.0.5
```

## Google Cloud Speech-to-Text API (Optional)

To enable cloud-based speech recognition:
1. Obtain a service account JSON key with the Speech API enabled.
2. Place it under the `Demo/` folder as `service-account.json`.
3. For more information, visit the [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text/) page.

```json
{
  "type": "service_account",
  "project_id": "",
  "private_key_id": "",
  "private_key": "",
  "client_email": "",
  "client_id": "",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "",
  "universe_domain": "googleapis.com"
}

```

---

### 3. Virtual Speaker and Mic Setup
For audio input:
```bash
pactl load-module module-null-sink sink_name="virtual_speaker" sink_properties=device.description="virtual_speaker"
pactl load-module module-remap-source master="virtual_speaker.monitor" source_name="virtual_mic" source_properties=device.description="virtual_mic"
```

Set the `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/alsa-lib/:$LD_LIBRARY_PATH
```

---

## 4. Model Setup

### EMS-Whisper
1. Download the `models` folder from [here](https://drive.google.com/drive/folders/1Z4oakBCSiSyW_agq3UG1eHsQDMmHKHOA?usp=sharing) and place it under `Demo/EMS_Whisper/models/`.
2. Go to directory `Demo/EMS_Whisper`.
3. Execute following commands to compile Whisper_CPP. Make sure to update `CUDA_ARCH_FLAG` in the `MakeFile` if `CUBLAS` is used (recommended).
```bash
make clean
WHISPER_CUBLAS=1 make -j
```
4. Verify if `stream` artifact is generated within the folder.

### EMS-TinyBERT
1. Download the `models` folder from [here](https://drive.google.com/drive/folders/1y4Ko6iSr5zkmYbm2llNQq7cwL-cu6Qi3?usp=sharing) and place it under `Demo/EMSAgent/Interface/models/`.


### EMS-Vision
1. Download the `models` folder from [here](https://drive.google.com/drive/folders/1y4Ko6iSr5zkmYbm2llNQq7cwL-cu6Qi3?usp=sharing) and place it under `Demo/EMSVision/weights/`.


---

### 5. Android Application Setup

To install and configure the **CognitiveEMS** Android application on the smartglass, follow the steps below:

#### **1. Open the Project in Android Studio**
- Launch **Android Studio** and open the project located at:  
  ```
  AndroidDevelopment/cognitive_ems
  ```
- Ensure you have the correct **JDK** and **SDKs** installed for Android development.

#### **2. Sync Gradle and Resolve Dependencies**
- Allow **Gradle** to synchronize and resolve dependencies.  
- Ensure there are no synchronization issues.

#### **3. Connect the Smartglass**
- Use a **USB cable** to connect the smartglass to your development machine.
- Enable **Developer Options** on the smartglass.
- Ensure **USB Debugging** is turned on.

#### **4. Verify Device Connection**
- Open **Android Studio** and check that the smartglass appears under **Connected Devices**.

#### **5. Configure Network Settings (Important ⚠️)**
- Ensure the **smartglass** and the **server hosting the CognitiveEMS pipeline** are on the **same network**.
- Determine the **server's IP address**.  
  - Ideally, assign a **static IP** to the server to avoid connectivity issues.
- Open the file:  
  ```
  AndroidDevelopment/cognitive_ems/app/src/main/res/values/strings.xml
  ```
- Update the following line with the **server's IP address**:  
  ```xml
  <string name="server_ip">YOUR_SERVER_IP_HERE</string>
  ```
  Example:
  ```xml
  <string name="server_ip">192.168.1.100</string>
  ```

![Android Studio](Assets/Android_Studio.png)


#### **6. Build and Deploy the Application**
- Compile the project in **Android Studio**.
- Install the application on the **smartglass**.

---



## Usage

### Running the Cognitive Assistant Server

1. Navigate to EMS-Pipeline directory.
2. Activate the conda environment.
3. Go to `Demo` folder and execute ```python GUI.py```. This should open up the main GUI for the cognitive assistant.
![Main GUI](Assets/Main_GUI.png)


4. To run the pipeline with pre-recorded audio transcripts, select the drop down `Microphone` and select one of the files. (Important ⚠️: Only speech recognitio and protocol prediction will work under this setting)
5. To toggle between Google Speech (Internet connectivity required) and Local Speech Model (Whisper) simply click the appropriate radio button.
6. To run the pipeline, press `Start`.

### Running the SocketIO Server

1. To enable communication between the Cognitive Assistant and the smartglass application, socketio server needs to run in the background (Important ⚠️).
2. Navigate to `EMS-Pipeline/Demo` folder.
3. Activate the `Conda` environment.
4. Run the server by executing `python flask_socket_server.py` and make sure it runs in the background.


### Running the AR Smartglass Application

1. In the menu screen of the smartglass, open the application with the name `CognitveEMS`.
2. Make sure that requested permissions from the app (mic,camera,network) is allowed and restart the application.
3. There is no interaction with the application and if above steps with Cognitive Assistant and SocketIO server is properly executed, the app will communicate with the Cognitive Assistant.
4. (Important ⚠️) The application may not be comlpetely optimized to use system resources. This was part of R&D and please use with that in mind. Application may close by itself (rarely) and may not maintain the connection with socket server over a long time (1+ hours).

### Smartwatch Integration
*In progress*


### Demonstrating the system

1. Once all above steps are executed and Cognitive Assistant is running, Smartglass application is running, you may test the system using following instructions.
2. To begin speech recognition, make sure the Microphone radio button is pressed and start speaking to the smartglass. When Google speech is used, you should see immediate speech translation. THe protocol model will process the speech when it detects a pause in your speech. If Whisper model is used, make sure in the OS sound settings, output is selected to be virtual_speaker and input is virtual_mic.
3. After a test, stop the speech recognition by pressing `stop` button in the main GUI. To clear the current speech transcript and protocol predictions, press `reset`. To start again, follow the above steps.

---


