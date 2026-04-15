# CognitiveEMS Pipeline

## About This Project

![CognitiveEMS](Assets/CognitiveEMS_Arch.png)

The **CognitiveEMS Pipeline** is a decision support system designed for real-time, multimodal, and edge-deployed **Cognitive Assistant Systems for Emergency Medical Services (EMS)**. Its primary goal is to enhance situational awareness for first responders and paramedics by automatically collecting and analyzing multimodal data from incident scenes, providing actionable insights and suggestions. The system comprises advanced AI models and modular software components designed for deployment on edge devices.

For more information, visit the [project page](https://www.nist.gov/ctl/pscr/cognitive-assistant-systems-emergency-response).

---

## Demo Overview

This branch contains a **portable Ubuntu-based demo** of the CognitiveEMS pipeline with a desktop GUI and device integrations.

### Server Features
- Real-time **speech recognition**
- **Protocol prediction** based on running transcript context
- **EMS object detection**
- Preliminary **EMS intervention/activity detection**

### AR Smartglass Features
- **Augmented Reality (AR)** feedback displaying:
  - Predicted protocols
  - Detected EMS objects
  - Detected interventions
- Real-time streaming of egocentric video and audio

### Smartwatch Features
- Real-time streaming of **hand movement data**

---

## High-Level Architecture

The current demo is split across a few major runtime components:

1. **Desktop GUI**  
   Entry point: `Demo/main.py`  
   This is the main operator-facing application.

2. **Speech Recognition**
   - **Google Cloud Speech-to-Text** for cloud inference
   - **EMS-Whisper / Whisper C++ streaming** for local inference

3. **Protocol Prediction**
   - **EMS-TinyBERT** consumes the running transcript and predicts likely EMS protocols

4. **Vision Inference**
   - The code under `Demo/EMS_Vision/` does **not** run the heavy vision models directly
   - Instead, it sends video frames to a separate **Docker-based inference server**
   - That server is documented in [Tools/EMS_Vision/README_container_inference.md](Tools/EMS_Vision/README_container_inference.md)

5. **Device Ingestion**
   - `simulator` source: incoming SRT stream
   - `smartglass` source: GUI-managed WebRTC server for the smartglass client

---

## Repository Layout

The directories most users will need are:

- `Demo/main.py`: desktop app entry point
- `Demo/GUI/`: main Qt GUI
- `Demo/EMS_Speech/`: speech recognition components
- `Demo/EMS_Agent/`: protocol prediction components
- `Demo/EMS_Vision/`: desktop-side vision client that talks to the Docker inference server
- `Demo/IO/`: SRT/WebRTC ingestion and local runtime I/O
- `Tools/EMS_Vision/`: Dockerized vision inference server, TensorRT conversion tools, and container guide
- `Android/EgoStreamer/`: Android smartglass submodule
- `Android/EgoStreamer/Android/EgoStreamer/`: Android Studio project for the smartglass app

---

## Hardware Requirements

### Recommended Specifications

#### Server
- **Processor**: 12th Gen (or newer) Intel Core i5, i7, or i9
- **RAM**: 32GB
- **Graphics**: NVIDIA RTX 3080 or higher

#### Smartglass
- **Model**: [Vuzix M4000 Smart Glasses](https://www.vuzix.com/products/m4000-smart-glasses)

#### Smartwatch
- **Model**: Samsung Galaxy Watch 5

---

### Minimum Specifications

#### Server
- **Processor**: 12th Gen Intel Core i5
- **RAM**: 16GB
- **Graphics**: NVIDIA RTX 3060

---

## Software Requirements

#### Server
- **OS**: Ubuntu 22.04.5 LTS (64-bit)
- **CUDA**: Version 12.x recommended for GPU-backed workloads
- **Environment**: Conda with Python 3.10
- **Docker + NVIDIA Container Toolkit**: required for the current EMS-Vision runtime
- **Android Studio**: for Android app deployment to smartglass and smartwatch

#### Smartglass
- **OS**: Android OS

#### Smartwatch
- **OS**: Android Wear OS

---

## Prerequisites

### Network Configuration
- Ensure the server, smartglass, and smartwatch are on the **same network**
- Assign a **static IP** to the server and configure it in the Android applications where needed
- Remove any firewall rules blocking required TCP or UDP traffic

![Network Architecture](Assets/Network_Architecture.png)

---

## Installation

> The setup steps below reflect the current directory structure and runtime flow. Some components are still evolving, but these are the correct starting points for the present demo.

### 1. Clone the Repository

Clone with submodules so the Whisper runtime code is available immediately:

```bash
git clone --recurse-submodules https://github.com/UVA-DSA/EMS-Pipeline.git
cd EMS-Pipeline
git checkout demo
git submodule update --init --recursive
```

If you already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create the Conda Environment

```bash
conda create -n EMSProject python=3.10
conda activate EMSProject
```

### 3. Install System Packages

Install the desktop, audio, SRT, and build dependencies:

```bash
sudo apt update
sudo apt upgrade

sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  ffmpeg \
  libav-tools \
  libsdl2-dev \
  libasound-dev \
  portaudio19-dev \
  libportaudio2 \
  libportaudiocpp0 \
  libsrt1.5-gnutls \
  libsrt-gnutls-dev \
  libxcb-randr0-dev \
  libxcb-xtest0-dev \
  libxcb-xinerama0-dev \
  libxcb-shape0-dev \
  libxcb-xkb-dev
```

### 4. Install Python Packages

Install PyTorch first for your CUDA version. The example below matches the currently tested setup:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.0+cu118.html
pip install -q torch-geometric==2.3.1
```

Then install the remaining application dependencies:

```bash
pip install \
  pyyaml \
  transformers==4.27.2 \
  pyqt5==5.15.6 \
  pandas \
  openpyxl \
  evaluate \
  jiwer \
  pyaudio \
  nltk \
  python-socketio \
  google-cloud-speech \
  pygame \
  sounddevice \
  mediapipe \
  netifaces \
  opencv-python-headless \
  py-trees==2.0.5 \
  requests \
  aiohttp \
  aiortc \
  qrcode[pil]
```

### 5. Speech Recognition Setup

The demo supports both cloud and local speech recognition.

#### Google Cloud Speech-to-Text API (Optional)

To enable cloud-based speech recognition:

1. Obtain a service account JSON key with the Speech API enabled
2. Place it in the `Demo/` folder as `service-account.json`
3. The GUI will automatically pick it up from:

```text
Demo/service-account.json
```

Reference:
- [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text/)

Expected JSON structure:

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

#### Local Whisper Setup

The current local speech path uses the Whisper submodule at:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream
```

Download the Whisper model files and place them under:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream/models
```

The current Python code expects the local model file:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream/models/ggml-finetuned-base-v203.bin
```

Build the realtime streaming binary:

```bash
cd Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream

cmake -B build --fresh \
  -DWHISPER_SDL2=ON \
  -DGGML_CUDA=1 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.2

cmake --build build -j --config Release
```

After the build, verify that this file exists:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream/build/bin/egosim_stream
```

If your system requires ALSA plugin discovery help, set:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/alsa-lib/:$LD_LIBRARY_PATH
```

### 6. Protocol Prediction Model Setup

Download the protocol prediction model files and place the extracted `models` directory under:

```text
Demo/EMS_Agent/Interface/models/
```

The current protocol runtime loads from:

```text
Demo/EMS_Agent/Interface/models/DKEC-TinyClinicalBERT/model.pt
```

### 7. Vision Setup

The current **EMS-Vision** runtime is containerized.

Important:
- The code under `Demo/EMS_Vision/` is a **client**
- It sends frames to a local inference server at `http://localhost:8000`
- It expects the Docker inference server described in [Tools/EMS_Vision/README_container_inference.md](Tools/EMS_Vision/README_container_inference.md) to be running first

Recommended setup path:

1. Follow the container guide in [Tools/EMS_Vision/README_container_inference.md](Tools/EMS_Vision/README_container_inference.md)
2. First try the **prebuilt Docker image**
3. If the prebuilt image or TensorRT engines are not compatible with your GPU architecture, follow the same guide to **build from source on the target machine**

This matters because TensorRT engines are GPU-family sensitive. In practice, that means:
- prebuilt images are the fastest way to get started
- some machines will still need a local rebuild for correct TensorRT compatibility

Before launching the full demo, verify the inference server is healthy:

```bash
curl http://localhost:8000/health
```

If you want object detection and intervention/activity inference in the GUI, keep that container running while using the demo.

---

### 8. Android Smartglass Application Setup

The current smartglass application is stored in the `Android/EgoStreamer` git submodule.

If you cloned the main repository without submodules, initialize it first:

```bash
git submodule update --init --recursive
```

At a minimum, make sure this submodule is present:

```text
Android/EgoStreamer
```

To install and use the smartglass application:

#### **1. Open the Android Project in Android Studio**

- Launch **Android Studio**
- Open the project located at:
  ```text
  Android/EgoStreamer/Android/EgoStreamer
  ```
- Ensure the Android SDK and Gradle dependencies are allowed to sync

#### **2. Connect the Smartglass**

- Use a **USB cable** to connect the smartglass to your development machine
- Enable **Developer Options** on the smartglass
- Enable **USB Debugging**
- Confirm the device appears in Android Studio under **Connected Devices**

#### **3. Build and Install the App**

- Build the project in Android Studio
- Install the app on the smartglass

The Android app requests and uses:
- camera access
- microphone access
- network access

#### **4. Put the Smartglass and Server on the Same Network**

- Ensure the **smartglass** and the **desktop system running the CognitiveEMS GUI** are on the **same network**
- A stable local network is strongly recommended for the current WebRTC-based setup

#### **5. Start the Desktop GUI in Smartglass Mode**

On the desktop side:

1. Launch the CognitiveEMS GUI:
   ```bash
   cd Demo
   python main.py
   ```
2. Select `smartglass` as the data source
3. Press `Start`

In the current code path, the GUI starts its local WebRTC smartglass server and displays a **QR code / WebSocket connection URL** in the video panel.

#### **6. Connect the Smartglass App**

On the smartglass:

1. Open the installed app
2. Grant camera and microphone permissions if prompted
3. Tap **Scan QR**
4. Scan the QR code displayed by the desktop GUI
5. Tap **Start Streaming**

The Android app expects a WebSocket signaling URL of the form:

```text
ws://<server-ip>:<port>/ws
```

In normal use, you should rely on the QR code so the correct host and port are captured automatically.

#### **7. Manual URL Entry Fallback**

If QR scanning fails, the app also allows manual entry of the WebSocket URL.

In that case:
- use the exact `ws://.../ws` URL shown by the desktop GUI
- make sure the smartglass can reach that host on the local network

![Android Studio](Assets/Android_Studio.png)

---

## Usage

### Quick Start

For the current end-to-end demo, use this order:

1. Activate your environment
2. Start the EMS-Vision Docker inference server and verify `http://localhost:8000/health`
3. Launch the desktop GUI with `python main.py` from the `Demo/` directory
4. Choose `simulator` or `smartglass` as the source
5. Choose the speech backend:
   - `Google Speech Cloud Model`
   - `OpenAI Whisper Local Model`
6. Press `Start`

### Running the Cognitive Assistant GUI

1. Navigate to the repository root
2. Activate the conda environment
3. Start the GUI from the `Demo` directory:

```bash
cd Demo
python main.py
```

This launches the current desktop application entry point.

![Main GUI](Assets/Main_GUI.png)

### Source Selection in the GUI

The current GUI supports two source modes:

#### 1. `simulator`

Use this when an upstream simulator is sending a multimodal SRT stream to the desktop application.

You will need to configure:
- **Height**
- **Width**
- **IP Address**
- **Port**

Notes:
- The desktop app uses `Demo/IO/srt_receiver.py` for this path
- The host machine must have `libsrt` installed
- The sender must already be streaming to the configured host and port

#### 2. `smartglass`

Use this when connecting the smartglass client directly to the desktop app.

In the current code path:
- selecting `smartglass` and pressing `Start` causes the GUI to launch its local WebRTC ingestion server
- the server is started from `Demo/IO/webrtc_server.py`
- the GUI then displays a **connection URL / QR code** in the video panel

Important:
- There is **no separate `flask_socket_server.py` step** in the current desktop runtime
- The older `python GUI.py` and Socket.IO instructions no longer match the present code path

### Speech Backend Selection

The GUI currently exposes two speech modes:

#### Google Speech Cloud Model

Use this when:
- internet connectivity is available
- `Demo/service-account.json` is present
- you want cloud speech recognition

#### OpenAI Whisper Local Model

Use this when:
- you have built `egosim_stream`
- the Whisper model file is available locally
- you want offline/local speech recognition

### Vision Runtime Behavior

For the current demo:

- the desktop app posts frames to the local inference container at `http://localhost:8000`
- object detection requests go to `/infer/detr`
- activity/intervention requests go to `/infer/activity/{stream_id}`

If that container is not running, vision features will not work correctly.

### Running the AR Smartglass Application

1. Make sure the `Android/EgoStreamer` submodule has been cloned and the app has been installed on the smartglass
2. Ensure the smartglass and the desktop machine are on the same local network
3. Start the desktop GUI and select `smartglass` as the source
4. Press `Start` in the desktop GUI so the local smartglass WebRTC server starts and shows a QR code
5. Open the smartglass app
6. Grant camera and microphone permissions if prompted
7. Tap **Scan QR** and scan the QR code shown by the desktop GUI
8. Tap **Start Streaming**
9. If scanning fails, manually enter the exact `ws://.../ws` URL shown by the GUI

### Smartwatch Integration

*In progress*

While the simulator supports smartwatch data streaming, full pipeline is not yet implemented with an actual smartwatch wirelessly streaming data in realtime.

### Demonstrating the System

1. Complete the required setup for speech, protocol prediction, and vision
2. Start the Docker inference server for vision if you want detections and interventions
3. Launch the GUI with `python main.py`
4. Select the desired source and speech backend
5. Press `Start`
6. Confirm the following update in the GUI:
   - transcript text
   - protocol predictions
   - vision detections / activities
7. Press `Stop` when the session is complete

---

## Troubleshooting

### `egosim_stream` not found

Make sure you:
- initialized submodules
- built `Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream`
- verified the output binary exists at `build/bin/egosim_stream`

### Whisper model not found

Make sure this file exists:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream/models/ggml-finetuned-base-v203.bin
```

### Vision container is not reachable

Check:

```bash
curl http://localhost:8000/health
```

If this fails, start or rebuild the Docker inference server using:

- [Tools/EMS_Vision/README_container_inference.md](Tools/EMS_Vision/README_container_inference.md)

### SRT simulator mode fails to connect

Check:
- the simulator is actually streaming to the configured host and port
- `libsrt` is installed on the host
- the configured resolution matches the sender
- local firewall rules are not blocking the connection

### Google speech is unavailable in the GUI

Check:
- internet connectivity
- `google-cloud-speech` is installed
- `Demo/service-account.json` exists and is valid

---
