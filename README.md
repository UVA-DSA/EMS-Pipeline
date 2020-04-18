# RAA Style Transfer

This branch contains files on converting medical transcripts to ones that resemble human-spoken transcripts.

Transcripts should be placed in the first column of data/transcripts.csv. Outputted converted transcripts will show up in the same file.

Tested and verified on Python 3.7.4.

# Instructions

1. Ensure that all transcripts to be converted are located in the first column of `data/transcripts.csv`.
2. Create new virtual environment: `python -m venv style_venv`
3. Activate virtual environment: On Windows: `style_venv\Scripts\activate.bat` On Linux: `source style_venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Run: `python style_transfer.py`
