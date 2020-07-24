# pyConTextNLP concept extractor (code used in pipeline integration)

This repository contains the code that performs concept extraction and negation detection on sentences from a speech-to-text engine. It is integrated as part of https://github.com/UVA-DSA/EMS-Pipeline.

An adaptation of this code is called by the pipeline, passed a sentence, and returns any concepts that are detected along with their negation status'. The code in this branch prompts the user for a sentence (which must be ended with puncutation such as a period), and prints out any concept CUIs detected. The printed output corresponds to what the pipeline would use to perform protocol modeling.

# Installation and run instructions

Tested to be working on Python 3.7.4.

1. Create new Python virtual environment:
	- python -m venv negation_venv
2. Activate new Python virtual environment:
	- ON LINUX:
		- source negation_venv/bin/activate
	- ON WINDOWS:
		- negation_venv\Scripts\activate.bat
3. Install pip requirements:
	- pip install -r requirements.txt
4. Run the program:
	- python OneSentenceNegationDetector.py
