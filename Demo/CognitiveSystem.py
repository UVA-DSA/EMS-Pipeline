from __future__ import absolute_import, division, print_function
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
import commands
import datetime
import re

# ============== Cognitive System ==============

# NLP Thread
def CognitiveSystem(Window, SpeechToNLPQueue):
    # Create signal objects
    SpeechSignal = GUISignal()
    SpeechSignal.signal.connect(Window.UpdateSpeechBox)

    NLPSignal = GUISignal()
    NLPSignal.signal.connect(Window.UpdateNLPBox)

    # BT framework parameters
    execfile("bt_parameters.py")

    # Setup BT Framework
    blackboard = Blackboard()
    root = py_trees.composites.Sequence("Root_1")
    IG = be.InformationGathering()
    TC = be.TextCollection()
    V = be.Vectorize()
    PS = be.ProtocolSelector()
    root.add_children([TC,IG,V,PS,be.protocols])
    behaviour_tree = py_trees.trees.BehaviourTree(root)
    behaviour_tree.add_pre_tick_handler(pre_tick_handler)
    behaviour_tree.setup(15)
    #py_trees.display.render_dot_tree(root)
    NLP_Dict = dict()
    SpeechText = ""
    NLP_Items = []
    Tick_Counter = 1

    while True:
        if threading.current_thread().stopped():
            print('NLP Thread Killed.')
            break

        # Get queue item from the Speech-to-Text Module
        received = SpeechToNLPQueue.get()
        print("Received chunk")

        if(received == 'Kill'):
            print("NLP Thread received Kill Signal")
            threading.current_thread().stop()

        # If item received from queue is legitmate
        else:
            #sentsList = [received.transcript]
            
            # Use online tool to find sentence boundaries
            dummy12= received.transcript
            dummy12 = dummy12.replace('\r', '').replace('\n', '')
            dummyP2=dummy12.replace(' ','%20')
            dummyP3=dummyP2.replace('\'','%27')
            dummyP=dummyP3.replace('&','%26')
            part1='curl -d text='+dummyP+' http://bark.phon.ioc.ee/punctuator'
            op = commands.getstatusoutput(part1)
            output = op[1].rsplit('\n', 1)[1]
            sentsList = textParse2.sent_tokenize(output) #final sentences

            PunctuatedAndHighlightedText = ""

            # Processes each chunk/sentence
            for idx, item in enumerate(sentsList):

                blackboard.text = [item]
                behaviour_tree.tick_tock(
                    sleep_ms=50,
                    number_of_iterations=1,
                    pre_tick_handler=None,
                    post_tick_handler=None)

                pr, sv_s, s = TickResults(Window, NLP_Items)

                PunctuatedAndHighlightedTextChunk = item

                for sv in sv_s:
                    if(sv[5] == Tick_Counter): # if new concept found in this tick
                        try:
                            i = re.search(r'%s' % sv[3], PunctuatedAndHighlightedTextChunk).start()
                            PunctuatedAndHighlightedTextChunk = str(PunctuatedAndHighlightedTextChunk[:i] + '<font color="red">' + PunctuatedAndHighlightedTextChunk[i:i + len(sv[3])] + '</font>' + PunctuatedAndHighlightedTextChunk[i + len(sv[3]):])   #   
                        except Exception as e:
                            pass

                PunctuatedAndHighlightedText += PunctuatedAndHighlightedTextChunk + " "
                Tick_Counter += 1

            PunctuatedAndHighlightedText = '<b>' + PunctuatedAndHighlightedText + '</b>'
            SpeechSignal.signal.emit([SpeechNLPItem(PunctuatedAndHighlightedText, received.isFinal, received.confidence, received.numPrinted, 'NLP')])

            
# Function to return this recent tick's results
def TickResults(Window, NLP_Text):

    NLPSignal = GUISignal()
    NLPSignal.signal.connect(Window.UpdateNLPBox)

    ProtocolSignal = GUISignal()
    ProtocolSignal.signal.connect(Window.UpdateProtocolBoxes)

    b = Blackboard()
    protocol_candidates = []
    signs_and_vitals = []
    suggestions = []

    print("===============================================================")

    #======= Top 3 protocol candidates
    print("\n======= Top 3 protocol candidates:")
    for p in b.protocol_flag:
        print(p, b.protocol_flag[p])
        binary = b.protocol_flag[p][0]
        confidence = b.protocol_flag[p][1]
        if(binary):
            try:
                if(confidence != 'nan' and float(confidence) > 0.0):
                    protocol_candidates.append((str(p), confidence))
            except Exception as e:
                pass

    # Sort by confidence and take top 3
    protocol_candidates = sorted(protocol_candidates, key = itemgetter(1), reverse = True)[:3]

    #======= Signs, symptoms, and vitals
    print("\n======= Signs, symptoms, and vitals:")

    for item in b.Vitals:
        if len(b.Vitals[item].content) > 0:
            content = (str(b.Vitals[item].name).capitalize(), str(b.Vitals[item].binary),
            str(b.Vitals[item].value), str(b.Vitals[item].content), 
            str(round(b.Vitals[item].score/1000, 2)), b.Vitals[item].tick)
            print(content)
            signs_and_vitals.append(content)
            if(content not in NLP_Text):
                NLP_Text.append(content)
    
    for item in b.Signs:
        if len(b.Signs[item].content) > 0:
            content = (str(b.Signs[item].name).capitalize(), str(b.Signs[item].binary),
            str(b.Signs[item].value), str(b.Signs[item].content),
            str(round(b.Signs[item].score/1000, 2)), b.Signs[item].tick)
            print(content)
            signs_and_vitals.append(content)
            if(content not in NLP_Text):
                NLP_Text.append(content)

    # Sort by Tick
    signs_and_vitals = sorted(signs_and_vitals, key = itemgetter(5))

    #======= Suggestions
    print("\n======= Suggestions:")
    for key in b.feedback:
        if b.feedback[key] > 0.1:
            content = (str(key).capitalize(), str(round(b.feedback[key], 2)))
            suggestions.append(content)
            print(content)

    # Sort by Concept
    suggestions = sorted(suggestions, key = itemgetter(1), reverse = True)

    #========================== Create output strings formatted for readibility
    signs_and_vitals_str = ""
    protocol_candidates_str = ""
    for i, p in enumerate(protocol_candidates):
        protocol_candidates_str += "(" + p[0] + ", <b>" + str(round(p[1], 2)) + "</b>)<br>"

    #for sv in signs_and_vitals:
    #    signs_and_vitals_str += "("
    #    for i, t in enumerate(sv):
    #        signs_and_vitals_str += str(t)[0:len(t)] + ", "
    #    signs_and_vitals_str =  signs_and_vitals_str[:-2] + ")\n"

    for sv in NLP_Text:
        signs_and_vitals_str += "("
        for i, t in enumerate(sv):
            if(i != 3 and i != 4 and i != 5):
                signs_and_vitals_str += str(t)[0:len(str(t))] + ", "
            if(i == 4):
                signs_and_vitals_str += "<b>" + str(t)[0:len(str(t))] + "</b>, "
        signs_and_vitals_str =  signs_and_vitals_str[:-2] + ")<br>"
    
    suggestions_str = ""
    for s in suggestions:
        suggestions_str += "(" + str(s[0]) + ", <b>" + str(s[1]) + "</b>)<br>"

    print("===============================================================")

    ProtocolSignal.signal.emit([protocol_candidates_str, suggestions_str])
    NLPSignal.signal.emit([signs_and_vitals_str])

    return protocol_candidates, signs_and_vitals, suggestions

# extract concept and calculate similarity
def pre_tick_handler(behaviour_tree):
    blackboard = Blackboard()
    blackboard.tick_num += 1

