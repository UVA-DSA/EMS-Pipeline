# Installation and Setup Guide

This guide provides instructions for setting up the **CognitiveEMS** Pipeline on *NVIDIA Jetson Nano* edge device.

## Disassemble and Install WiFi Card

[Insert instructions for disassembling and installing the WiFi card here.]

## Anaconda Compatibility

Anaconda is not supported on Arch Linux. To manage Python environments, we recommend using `archiconda3`.

Please follow the instructions at [archiconda3](https://github.com/yqlbu/archiconda3) to install Conda.

## Permissions Issue

If you encounter permission issues, you can resolve them by changing ownership of certain directories:

```bash
sudo chown -R $USER:$USER ~/archiconda3/
sudo chown -R $USER:$USER ~/.conda/
```

## PyTorch Compatibility

To ensure compatibility, use PyTorch 1.10 with Cuda 10.2 and Python 3.6. Be mindful when creating virtual environments.

Create a Conda environment with Python 3.6:


```bash
conda create -n myenv python=3.6
conda activate myenv
```

For Cuda 1.10.0, refer to [this link](https://qengineering.eu/install-pytorch-on-jetson-nano.html).

## Numpy Version

Ensure that you install numpy version 1.19.4 to avoid issues:

```bash
pip install numpy==1.19.4
```
## PyAudio Installation

Install the required dependencies and PyAudio:


```bash
sudo apt-get install portaudio19-dev python-all-dev python3-all-dev
pip install pyaudio
```

## Transformers Compatibility

Please note that the latest Transformers library may not be supported. You may encounter issues with the WhisperProcessor. Additionally, the 'evaluate' module is not supported.

## Protocol Agent Warmup Phase

The protocol agent requires a warmup phase before use. Follow these steps:

1.  Start the Whisper CPP process.
2.  Start the EMSAgent process.
3.  Execute a warmup inference inside EMSAgent to load kernels.
4.  Sleep for around 60 seconds for the warmup to complete.
5.  Start the audio streaming process.

## GPU Loading Time on Jetson

On Jetson devices, it may take a significant amount of time to load PyTorch kernels to the GPU. This typically happens only once.

----------

Feel free to customize and expand upon these instructions as needed for your specific setup and requirements.

