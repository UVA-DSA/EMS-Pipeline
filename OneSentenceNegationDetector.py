import pyConTextNLP.pyConText as pyConText
import pyConTextNLP.itemData as itemData
import networkx as nx
import os
import sys


def one_sentence_detection(test_str):
    # "pulse" concept CUI was used for "pulse rate"
    # "resp" concept CUI was used for "respiratory rate"

    dict_concept_to_cui = {
        "ABSENT_FEMORAL_PULSES": "C0239556",
        "AGITATION": "C0085631",
        "ANXIETY": "C0003467",
        "BIZARRE_BEHAVIOR": "C0474407",
        "BLINDNESS": "C0456909",
        "BLOOD_PRESSURE": "C0005823",
        "CHEST_PAIN": "C0008031",
        "CLAMMY_SKIN": "C0392162",
        "COMBATIVE": "C0241845",
        "CONFUSION": "C0009676",
        "CONSTIPATION": "C0009806",
        "COOL_SKIN": "C1997526",
        "COUGH": "C0010200",
        "DECREASED_MENTAL_STATUS": "C0746541",
        "DECREASED_VISUAL_ACUITY": "C0234632",
        "DELUSIONAL_THOUGHTS": "C0011253",
        "MALAISE": "C0231218",
        "DIAPHORESIS": "C0700590",
        "DIARRHEA": "C0011991",
        "DISTENSION": "C0012359",
        "DIZZINESS": "C0012833",
        "DYSRHYTHMIA": "C0003811",
        "DYSURIA": "C0013428",
        "FEVER": "C0015967",
        "HALLUCINATIONS": "C0018524",
        "HEADACHE": "C0018681",
        "HOMICIDAL_THOUGHTS": "C0455204",
        "HYPERTENSION": "C0020538",
        "HYPERTHERMIA": "C0015967",
        "HYPOTENSION": "C0020649",
        "HYPOTHERMIA": "C0020672",
        "INCONTINENCE": "C0021167",
        "LIGHTHEADEDNESS": "C0012833",
        "LOSS_OF_CONSCIOUSNESS": "C0041657",
        "MENTAL_STATUS_CHANGES": "C0856054",
        "MYALGIAS": "C0231528",
        "NAUSEA": "C0027497",
        "PAIN": "C0006138",
        "PAIN_RADIATION": "C1507012",
        "PAIN_REGION": "C0007859",
        "PAIN_SEVERITY": "C1507013",
        "PALE": "C0030232",
        "PALPITATIONS": "C0030252",
        "PREGNANCY": "C0032961",
        "PULSE_RATE": "C0034107",
        "RALES": "C0034642",
        "RASH": "C0015230",
        "RESPIRATORY_RATE": "C0231832",
        "RHONCHI": "C0034642",
        "RIGIDITY": "C0026837",
        "SEIZURE": "C0036572",
        "SHORTNESS_OF_BREATH": "C0013404",
        "SLEEPINESS": "C0013144",
        "SUICIDAL_THOUGHT": "C0424000",
        "TACHYCARDIA": "C0039231",
        "TENDERNESS": "C0234233",
        "TRAUMA": "C0043251",
        "TRIPODING": "C0454273",
        "UNEQUAL_FEMORAL_PULSES": "C1997431",
        "USE_OF_ACCESSORY_MUSCLES": "C0425468",
        "VAGINAL_BLEEDING": "C0566967",
        "VAGINAL_DISCHARGE": "C0227791",
        "VIOLENT": "C0242151",
        "VOMITING": "C0042963",
        "WEAKNESS": "C3714552",
        "WHEEZING": "C0043144"
    }

    PYCONTEXTNLP_MODIFIERS = r'/' + os.getcwd() + '/data/pycontextnlp_modifiers.yml'
    PYCONTEXTNLP_TARGETS = r'/' + os.getcwd() + '/data/pycontextnlp_targets.yml'

    modifiers = itemData.get_items(PYCONTEXTNLP_MODIFIERS)
    targets = itemData.get_items(PYCONTEXTNLP_TARGETS)

    # handle non-case here (nontender, non tender, non-tender)
    list_tokens = test_str.split()
    for i in range(len(list_tokens)):
        token = list_tokens[i]
        if token[0:3] == 'non':
            if len(token) == 3:
                # non tender
                potential_negated_concept = list_tokens[i + 1]
                not_string_to_test = "can be ruled out for " + potential_negated_concept + "."
                markup = pyConText.ConTextMarkup()
                markup.setRawText(not_string_to_test.lower())
                markup.cleanText()
                markup.markItems(modifiers, mode="modifier")
                markup.markItems(targets, mode="target")
                markup.pruneMarks()
                markup.applyModifiers()
                list_non_edges = markup.edges()
                for edge in list_non_edges:
                    if 'neg' in edge[0].getCategory()[0]:

                        # get position of both "non" and the concept being negated
                        list_positions_together = []
                        non_sum = 0
                        for j in range(i):
                            non_sum += len(list_tokens[j]) + 1
                        concept_start_char = non_sum + len(list_tokens[i]) + 1

                        concept_name = " ".join(edge[1].getCategory()[0].split('_')).lower()
                        negation_status = 1
                        concept_position_start = concept_start_char
                        concept_character_length = len(potential_negated_concept)

                        bp_object = []
                        bp_object.append('00000000')
                        bp_object.append('MMI')
                        bp_object.append('1')
                        bp_object.append(concept_name)

                        bp_object.append(dict_concept_to_cui[edge[1].getCategory()[0].upper()])

                        bp_object.append('x')
                        bp_object.append('[""-""-""-"' + concept_name + '"-""-' + str(negation_status) + ']')
                        bp_object.append('TX')
                        bp_object.append(str(concept_position_start) + '/' + str(concept_character_length))
                        bp_object.append('')

                        print(bp_object)

            elif token[3] == '-':
                # non-tender
                potential_negated_concept = token[4:]
                not_string_to_test = "can be ruled out for " + potential_negated_concept + "."
                markup = pyConText.ConTextMarkup()
                markup.setRawText(not_string_to_test.lower())
                markup.cleanText()
                markup.markItems(modifiers, mode="modifier")
                markup.markItems(targets, mode="target")
                markup.pruneMarks()
                markup.applyModifiers()
                list_non_edges = markup.edges()
                for edge in list_non_edges:
                    if 'neg' in edge[0].getCategory()[0]:
                        # get position of both "non" and the concept being negated
                        list_positions_together = []
                        non_sum = 0
                        for j in range(i):
                            non_sum += len(list_tokens[j]) + 1
                        concept_start_char = non_sum + 4

                        concept_name = " ".join(edge[1].getCategory()[0].split('_')).lower()
                        negation_status = 1
                        concept_position_start = concept_start_char
                        concept_character_length = len(potential_negated_concept)

                        bp_object = []
                        bp_object.append('00000000')
                        bp_object.append('MMI')
                        bp_object.append('1')
                        bp_object.append(concept_name)

                        bp_object.append(dict_concept_to_cui[edge[1].getCategory()[0].upper()])

                        bp_object.append('x')
                        bp_object.append('[""-""-""-"' + concept_name + '"-""-' + str(negation_status) + ']')
                        bp_object.append('TX')
                        bp_object.append(str(concept_position_start) + '/' + str(concept_character_length))
                        bp_object.append('')

                        print(bp_object)

            else:
                # nontender
                potential_negated_concept = token[3:]
                not_string_to_test = "can be ruled out for " + potential_negated_concept + "."
                markup = pyConText.ConTextMarkup()
                markup.setRawText(not_string_to_test.lower())
                markup.cleanText()
                markup.markItems(modifiers, mode="modifier")
                markup.markItems(targets, mode="target")
                markup.pruneMarks()
                markup.applyModifiers()
                list_non_edges = markup.edges()
                for edge in list_non_edges:
                    if 'neg' in edge[0].getCategory()[0]:
                        # get position of both "non" and the concept being negated
                        list_positions_together = []
                        non_sum = 0
                        for j in range(i):
                            non_sum += len(list_tokens[j]) + 1
                        concept_start_char = non_sum + 3

                        concept_name = " ".join(edge[1].getCategory()[0].split('_')).lower()
                        negation_status = 1
                        concept_position_start = concept_start_char
                        concept_character_length = len(potential_negated_concept)

                        bp_object = []
                        bp_object.append('00000000')
                        bp_object.append('MMI')
                        bp_object.append('1')
                        bp_object.append(concept_name)

                        bp_object.append(dict_concept_to_cui[edge[1].getCategory()[0].upper()])

                        bp_object.append('x')
                        bp_object.append('[""-""-""-"' + concept_name + '"-""-' + str(negation_status) + ']')
                        bp_object.append('TX')
                        bp_object.append(str(concept_position_start) + '/' + str(concept_character_length))
                        bp_object.append('')

                        print(bp_object)

    markup = pyConText.ConTextMarkup()
    markup.setRawText(test_str.lower())
    markup.cleanText()
    markup.markItems(modifiers, mode="modifier")
    markup.markItems(targets, mode="target")
    markup.pruneMarks()
    markup.applyModifiers()

    for edge in markup.edges():
        # [index, mm, score, preferred_name, CUI, semtypes, trigger, location, pos_info, tree_codes]

        # get total length of detection (first character of modifier/target to last character of modifier/target)
        list_positions = [edge[0].getSpan()[0], edge[0].getSpan()[1], edge[1].getSpan()[0], edge[1].getSpan()[1]]
        # ignore detections with a total length longer than a certain character count
        if max(list_positions) - min(list_positions) > 105:
            continue

        concept_name = " ".join(edge[1].getCategory()[0].split('_')).lower()
        negation_status = 0
        if 'neg' in edge[0].getCategory()[0]:
            negation_status = 1
        concept_position_start = edge[1].getSpan()[0]
        concept_character_length = edge[1].getSpan()[1] - concept_position_start

        bp_object = []
        bp_object.append('00000000')
        bp_object.append('MMI')
        bp_object.append('1')
        bp_object.append(concept_name)

        bp_object.append(dict_concept_to_cui[edge[1].getCategory()[0].upper()])

        bp_object.append('x')
        bp_object.append('[""-""-""-"' + concept_name + '"-""-' + str(negation_status) + ']')
        bp_object.append('TX')
        bp_object.append(str(concept_position_start) + '/' + str(concept_character_length))
        bp_object.append('')

        print(bp_object)


def main():
    while True:
        test_str = input("Please enter a sentence with punctuation at the end (example: Patient denies vomiting.):\n")
        one_sentence_detection(test_str)
        print("\n\n")


if __name__ == "__main__":
    main()