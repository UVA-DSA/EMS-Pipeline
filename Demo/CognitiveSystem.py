from __future__ import absolute_import, division, print_function
import os
from py_trees import blackboard
from six.moves import queue
import py_trees
import behaviours_m as be
from py_trees.blackboard import Blackboard
from tqdm import tqdm as tqdm
from classes import SpeechNLPItem, GUISignal
import threading
import text_clf_utils as utils
from ranking_func import rank
from Form_Filling import textParse2
from operator import itemgetter
import subprocess
import re

#added 3/18
import nltk
from nltk.data import find

# Function to check and download if necessary
def download_nltk_data(dataset_name):
    try:
        # Attempt to find the dataset
        if dataset_name == 'punkt':
            find(f"tokenizers/{dataset_name}")
        else:
            find(f"corpora/{dataset_name}")
            
        print(f"{dataset_name} is already available.")
    except LookupError:
        # If dataset is not found, download it
        print(f"Downloading {dataset_name}...")
        nltk.download(dataset_name, quiet=True)
#--
download_nltk_data('punkt')

# ============== Cognitive System ==============
from behaviours_m import blackboard
blackboard.tick_num = 0

# ------------ For Feedback ------------
class FeedbackObj:
    def __init__(self, intervention, protocol, p_confidence, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.protocol_confidence = p_confidence
        self.concept = concept

# ------------ End Feedback Obj Class ------------

# Cognitive System Thread
def CognitiveSystem(Window, SpeechToNLPQueue, FeedbackQueue, data_path_str, conceptBool, interventionBool):

    # Create GUI signal objects
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    ConceptExtractionSignal = GUISignal()
    ConceptExtractionSignal.signal.connect(Window.UpdateConceptExtractionBox)

    # Initialize BT framework parameters
    exec(open("./bt_parameters.py").read())

    # Setup BT Framework
    #blackboard = Blackboard()
    global blackboard
    root = py_trees.composites.Sequence("Root_1")
    IG = be.InformationGathering()
    TC = be.TextCollection()
    V = be.Vectorize()
    PS = be.ProtocolSelector()
    root.add_children([TC, IG, V, PS, be.protocols])  # be.protocols
    behaviour_tree = py_trees.trees.BehaviourTree(root)
    behaviour_tree.add_pre_tick_handler(pre_tick_handler)
    behaviour_tree.setup()
    Concepts_Graph = dict()
    SpeechText = ""
    NLP_Items = []
    Tick_Counter = 1

    previous_transcript = ""
    while True:
        # continue
        # Get queue item from the Speech-to-Text Module
        received = SpeechToNLPQueue.get()
        # print("Received chunk", received.transcript)

        if(received == 'Kill'):
            # print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
            break

        if(Window.reset == 1):
            # print("Cognitive System Thread Received reset signal. Killing Cognitive System Thread.")
            return

        # If item received from queue is legitmate
        else:
            #sentsList = [received.transcript]

            # Use online tool to find sentence boundaries
            current_transcript = received.transcript
            # dummy12 = dummy12.replace('\r', '').replace('\n', '')
            # dummyP2 = dummy12.replace(' ', '%20')
            # dummyP3 = dummyP2.replace('\'', '%27')
            # dummyP = dummyP3.replace('&', '%26')
            # part1 = 'curl -d text='+dummyP+' http://bark.phon.ioc.ee/punctuator'
            # op = subprocess.getstatusoutput(part1)
            # # print("op:  ", op)
            # output = op[1].rsplit('\n', 1)[1]
            # sentsList = textParse2.sent_tokenize(output)  # final sentences
            # print("original: ",received.transcript)
            # print("online_tool: ",output)
            # print("sentsList: ",sentsList)
            
            
            # # Split the strings into lists of words
            # words1 = previous_transcript.split()
            # words2 = current_transcript.split()

            # # Find the words in str2 that are not in str1 while preserving order
            # diff = [word for word in words2 if word not in words1]

            # # Join the words to form str3
            # new_transcript = ' '.join(diff)
            
            # if len(diff) == 0: pass 
            
            
            sentsList = [current_transcript]

            print("\n\n\n\ Concept Extraction Received: ",current_transcript, '\n\n')

            def print_tree(tree):
                print(py_trees.display.unicode_tree(root=tree.root, show_status=True))
                
                
            # print("sentsList:",sentsList)
            # Processes each chunk/sentence
            PunctuatedAndHighlightedText = ""
            for idx, item in enumerate(sentsList):
                
                blackboard.text = [item]
                behaviour_tree.tick_tock(
                    period_ms=50,
                    number_of_iterations=1,
                    pre_tick_handler=None)
                    # post_tick_handler=print_tree)

                pr, sv_s, s = TickResults(Window, NLP_Items, data_path_str, conceptBool, interventionBool, FeedbackQueue)

                PunctuatedAndHighlightedTextChunk = item

                for sv in sv_s:
                    if(sv[5] == Tick_Counter):  # if new concept found in this tick
                        try:
                            i = re.search(r'%s' % sv[3], PunctuatedAndHighlightedTextChunk).start()
                            PunctuatedAndHighlightedTextChunk = str(PunctuatedAndHighlightedTextChunk[:i] + '<span style="background-color: #FFFF00">' + PunctuatedAndHighlightedTextChunk[i:i + len(
                                sv[3])] + '</span>' + PunctuatedAndHighlightedTextChunk[i + len(sv[3]):])
                        except Exception as e:
                            pass

                PunctuatedAndHighlightedText += PunctuatedAndHighlightedTextChunk + " "
                Tick_Counter += 1

                NLP_Items = []

                if(Window.reset == 1):
                    # print("Cognitive System Thread Received reset signal. Killing Cognitive System Thread.")
                    return

            PunctuatedAndHighlightedText = '<b>' + PunctuatedAndHighlightedText + '</b>'
            # SpeechSignal.signal.emit([SpeechNLPItem(
            #     PunctuatedAndHighlightedText, received.isFinal, received.confidence, received.numPrinted, 'NLP')]) # icommented for somereaosn


# Function to return this recent tick's results
def TickResults(Window, NLP_Items, data_path_str, conceptBool, interventionBool, FeedbackQueue):
    if conceptBool == True:
        if not os.path.exists(data_path_str + "conceptextractiondata/"):
            os.makedirs(data_path_str+"conceptextractiondata/")
        CE_outputfile = open(data_path_str +"conceptextractiondata/"+ "cedata.txt", 'w')

    if interventionBool == True:
        if not os.path.exists(data_path_str + "interventiondata/"):
            os.makedirs(data_path_str+"interventiondata/")
        Intervention_outputfile = open(data_path_str +"interventiondata/" + "interventiondata.txt", 'w')

    ConceptExtractionSignal = GUISignal()
    ConceptExtractionSignal.signal.connect(Window.UpdateConceptExtractionBox)

    ProtocolSignal = GUISignal()
    ProtocolSignal.signal.connect(Window.UpdateProtocolBoxes)

    #blackboard = Blackboard()
    global blackboard
    protocol_candidates = []
    signs_and_vitals = []
    suggestions = []

    # print("===============================================================")

    # ======= Top 3 protocol candidates
    # print("\n======= Top 3 protocol candidates:")
    for p in blackboard.protocol_flag:
        # print(p, blackboard.protocol_flag[p])
        binary = blackboard.protocol_flag[p][0]
        confidence = blackboard.protocol_flag[p][1]
        if(binary):
            try:
                if(confidence != 'nan' and float(confidence) > 0.0):
                    protocol_candidates.append((str(p), confidence))
            except Exception as e:
                pass

    # Sort by confidence and take top 3
    protocol_candidates = sorted(protocol_candidates, key=itemgetter(1), reverse=True)[:3]

    # ======= Signs, symptoms, and vitals
    # print("\n======= Signs, symptoms, and vitals:")

    for item in blackboard.Vitals:
        if len(blackboard.Vitals[item].content) > 0:
            content = (str(blackboard.Vitals[item].name).capitalize(), str(blackboard.Vitals[item].binary),
                       str(blackboard.Vitals[item].value), str(blackboard.Vitals[item].content),
                       str(round(blackboard.Vitals[item].score/1000, 2)),blackboard.Vitals[item].tick)
            # print(content)
            signs_and_vitals.append(content)

            if(content not in NLP_Items):
                NLP_Items.append(content)

    for item in blackboard.Signs:
        if len(blackboard.Signs[item].content) > 0:
            content = (str(blackboard.Signs[item].name).capitalize(), str(blackboard.Signs[item].binary),
                       str(blackboard.Signs[item].value), str(blackboard.Signs[item].content),
                       str(round(blackboard.Signs[item].score/1000, 2)),blackboard.Signs[item].tick)
            # print(content)
            signs_and_vitals.append(content)
            if(content not in NLP_Items):
                NLP_Items.append(content)

    # Sort by Tick
    signs_and_vitals = sorted(signs_and_vitals, key=itemgetter(5))

    # ======= Suggestions
    # print("\n======= Suggestions:")
    for key in blackboard.feedback:
        if blackboard.feedback[key] > 0.1:
            content = (str(key).capitalize(), str(round(blackboard.feedback[key], 2)))
            suggestions.append(content)
            # print(content)

    # Sort by Concept
    suggestions = sorted(suggestions, key=itemgetter(1), reverse=True)

    # ========================== Create output strings formatted for readibility
    protocol_candidates_str = ""
    for i, p in enumerate(protocol_candidates):
        protocol_candidates_str += "(" + p[0] + ", <b>" + str(round(p[1], 2)) + "</b>)<br>"

    signs_and_vitals_str = ""
    signs_and_vitals_str_fb = ""

    for sv in NLP_Items:
        signs_and_vitals_str += "("
        for i, t in enumerate(sv):
            if(i != 3 and i != 4 and i != 5):
                signs_and_vitals_str += str(t)[0:len(str(t))] + ", "
                signs_and_vitals_str_fb += str(t)[0:len(str(t))] + ", "
            if(i == 4):
                signs_and_vitals_str += "<b>" + str(t)[0:len(str(t))] + "</b>, "
                signs_and_vitals_str_fb += str(t)[0:len(str(t))] + ", "

        signs_and_vitals_str = signs_and_vitals_str[:-2] + ")<br>"

    suggestions_str = ""
    suggestions_str_fb = ""
    for s in suggestions:
        suggestions_str += "(" + str(s[0]) + ", <b>" + str(s[1]) + "</b>)<br>"
        suggestions_str_fb += "(" + str(s[0]) + ", " + str(s[1]) + ")  | "

    # print("===============================================================")

    #Feedback
    interventionFB =  FeedbackObj(suggestions_str_fb, None, None, None)
    FeedbackQueue.put(interventionFB)

    conceptFB =  FeedbackObj(None,None,None, signs_and_vitals_str_fb)
    FeedbackQueue.put(conceptFB)


    # ProtocolSignal.signal.emit([protocol_candidates_str, suggestions_str])
    ProtocolSignal.signal.emit([interventionFB])


    if interventionBool == True:
        #write data to file for data collection
        Intervention_outputfile.write(suggestions_str)


    ConceptExtractionSignal.signal.emit([signs_and_vitals_str])

    if conceptBool == True:
        #write data to file for data collection
        CE_outputfile.write(signs_and_vitals_str)

    return protocol_candidates, signs_and_vitals, suggestions

# Extract concept and calculate similarity


def pre_tick_handler(behaviour_tree):
    #blackboard = Blackboard()
    global blackboard
    blackboard.tick_num += 1
