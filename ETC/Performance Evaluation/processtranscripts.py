#Elizabeth Shelton ejs6ar
"""
Converts a file of all transcripts into individual text files for each recording to be used with WERCalc.py; this should be run twice, once with the master list of original (ground truth) transcripts and once with the API results, and the filename should be modified below from "Orig" to "API" (or whatever you want, but change it in WERCalc.py too)

There will eventually be a function in WERCalc.py to do it with just the master list without generating so many files, for big sets of data
"""
def pull_files(masterlist, savepath):
    """
    :param: masterlist- a text file with all transcripts copied from Excel/Google Sheets
    :param: savepath- the path where these files should go
    """
    file = open(masterlist, "r")
    transcripts = []
    for thing in file:
        transcripts.append(thing)
    file.close()
    num_transcripts = len(transcripts)
    for i in range(num_transcripts):
        # for the second value, I used "Orig" for the original transcript and "API" for the text-to-speech result
        newfile = savepath + "Orig" + str(i) + ".txt"
        update = open(newfile, "w")
        print(transcripts[i], file=update)
        update.close()


# pull_files("C:/Users/Student/OneDrive/Documents/Summer 2019 Research/Week 2-Testing/Transcripts/masterlist.txt", "C:/Users/Student/OneDrive/Documents/Summer 2019 Research/Week 2-Testing/Transcripts/")

