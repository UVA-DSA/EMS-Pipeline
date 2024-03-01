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
nltk.download('punkt')
#--

# ============== Cognitive System ==============
from behaviours_m import blackboard
blackboard.tick_num = 0

# ------------ For Feedback ------------
class FeedbackObj:
    def __init__(self, intervention, protocol, concept):
        super(FeedbackObj, self).__init__()
        self.intervention = intervention
        self.protocol = protocol
        self.concept = concept

# ------------ End Feedback Obj Class ------------

# Cognitive System Thread
def CognitiveSystem(Window, TranscriptQueue, ProtocolQueue, data_path_str, conceptBool, interventionBool):

    # Create GUI signal objects
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateTranscriptBox)

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
# ================== HERE ================================================================
    while True:
        # continue
        # Get queue item from the Speech-to-Text Module
        received = TranscriptQueue.get()

        if(received == 'Kill'):
            print("Cognitive System Thread received Kill Signal. Killing Cognitive System Thread.")
            break

        if(Window.reset == 1):
            print("Cognitive System Thread Received reset signal. Killing Cognitive System Thread.")
            return

        # If item received from queue is legitmate
        else:
            #sentsList = [received.transcript]

            # Use online tool to find sentence boundaries
            dummy12 = received.transcript
            sentsList = [received.transcript]
            
            def print_tree(tree):
                print(py_trees.display.unicode_tree(root=tree.root, show_status=True))
                
            # Processes each chunk/sentence
            PunctuatedAndHighlightedText = ""
            for idx, item in enumerate(sentsList):
                
                blackboard.text = [item]
                behaviour_tree.tick_tock(
                    period_ms=50,
                    number_of_iterations=1,
                    pre_tick_handler=None)
                    # post_tick_handler=print_tree)

                pr, sv_s, s = TickResults(Window, NLP_Items, data_path_str, conceptBool, interventionBool, ProtocolQueue)

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

                if(Window.reset == 1):
                    # print("Cognitive System Thread Received reset signal. Killing Cognitive System Thread.")
                    return

            SpeechSignal.signal.emit([SpeechNLPItem(
                PunctuatedAndHighlightedText, received.isFinal, received.confidence, received.numPrinted, 'NLP')])


# Function to return this recent tick's results
def TickResults(Window, NLP_Items, data_path_str, conceptBool, interventionBool, ProtocolQueue):
    Concepts_Graph = dict()
    # print(NLP_Items)
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
                       str(round(blackboard.Vitals[item].score/1000, 2)), blackboard.Vitals[item].tick)
            # print(content)
            signs_and_vitals.append(content)
            if(content not in NLP_Items):
                NLP_Items.append(content)

    for item in blackboard.Signs:
        if len(blackboard.Signs[item].content) > 0:
            content = (str(blackboard.Signs[item].name).capitalize(), str(blackboard.Signs[item].binary),
                       str(blackboard.Signs[item].value), str(blackboard.Signs[item].content),
                       str(round(blackboard.Signs[item].score/1000, 2)), blackboard.Signs[item].tick)
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
    
    # Store in dictionary to take out duplicates of the same concept with out of date confidence
    for sv in NLP_Items:
        Concepts_Graph[sv[0]] = sv
    
    for sv in Concepts_Graph.values():
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
    interventionFB =  FeedbackObj(suggestions_str_fb, None, None)
    ProtocolQueue.put(interventionFB)

    conceptFB =  FeedbackObj(None,None, signs_and_vitals_str_fb)
    ProtocolQueue.put(conceptFB)


    # ProtocolSignal.signal.emit([protocol_candidates_str, suggestions_str])
    ProtocolSignal.signal.emit([protocol_candidates_str, suggestions_str])


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
