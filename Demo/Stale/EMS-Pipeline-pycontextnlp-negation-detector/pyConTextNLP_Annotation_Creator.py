import pyConTextNLP.pyConText as pyConText
import pyConTextNLP.itemData as itemData
import networkx as nx
import pandas as pd
import urllib
import os


def negations_pycontextnlp(clinical_text_df):
    for index, row in clinical_text_df.iterrows():
        # if the row has a NaN value in the transcripts column, skip it
        if (pd.isna(clinical_text_df.iloc[index, 0])):
            continue

        print("Transcript " + str(index) + ", row " + str(index + 2) + ":")

        # print("Detected negated edges:")
        list_detected_negated_edges, list_positions = negations_pycontextnlp_individual_transcript(row[0])

        print("Detected negated concepts:\n")

        # # UNCOMMENT THIS BLOCK TO DETECT ALL ANNOTATIONS (NOT JUST 'NON' CASE AND '-' CASE) ############
        #
        # for idx in range(len(list_detected_negated_edges)):
        #
        #     # handle opposite case
        #     if 'opposite' in list_detected_negated_edges[idx][1].getCategory()[0]:
        #
        #         list_positions_together = []
        #         for i in range(2):
        #             for j in range(2):
        #                 list_positions_together.append(list_positions[idx][i][j])
        #
        #         to_add = "".join(list_detected_negated_edges[idx][1].getCategory()[0].split('_opposite'))
        #         print("negated concept '" + to_add + "' detected at position ("
        #               + str(list_positions[idx][0][0]) + ", " + str(list_positions[idx][0][1])
        #               + ") (" + row[0][list_positions[idx][0][0]:list_positions[idx][0][1]] + "), ("
        #               + str(list_positions[idx][1][0]) + ", " + str(list_positions[idx][1][1]) + ") ("
        #               + row[0][list_positions[idx][1][0]:list_positions[idx][1][1]] + ")\n")
        #
        #         # print sentence being analyzed
        #         print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
        #         print("--------------------")
        #         print("..." + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100) : min(list_positions_together)]
        #             + '|||||'
        #             + row[0][min(list_positions_together):max(list_positions_together)]
        #             + '|||||'
        #             + row[0][max(list_positions_together) : (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)] + "...")
        #
        #         correct = input("\nIs this annotation correct? [y]es [n]o: ")
        #         if correct != 'n':
        #             annotation_string = '(' \
        #                                 + "".join(list_detected_negated_edges[idx][1].getCategory()[0]) \
        #                                 + ',False,' + row[0][list_positions[idx][1][0]:list_positions[idx][1][1]] \
        #                                 + ',' + row[0][list_positions[idx][0][0]:list_positions[idx][0][1]] \
        #                                 + ',' + str(min(list_positions_together)) \
        #                                 + ',' + str(max(list_positions_together)) \
        #                                 + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
        #                                 + ')'
        #             clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
        #             print('Added ' + annotation_string + ' to list of annotations.')
        #         print('\n')
        #
        #     # handle negative edge case
        #     elif 'neg' in list_detected_negated_edges[idx][0].getCategory()[0]:
        #
        #         list_positions_together = []
        #         for i in range(2):
        #             for j in range(2):
        #                 list_positions_together.append(list_positions[idx][i][j])
        #
        #         to_add = "".join(list_detected_negated_edges[idx][1].getCategory()[0].split('_'))
        #         print("negated concept '" + to_add + "' detected at position ("
        #               + str(list_positions[idx][0][0]) + ", " + str(list_positions[idx][0][1]) + ") ("
        #               + row[0][list_positions[idx][0][0]:list_positions[idx][0][1]] + "), ("
        #               + str(list_positions[idx][1][0]) + ", " + str(list_positions[idx][1][1]) + ") ("
        #               + row[0][list_positions[idx][1][0]:list_positions[idx][1][1]] + ")\n")
        #
        #         # print sentence being analyzed
        #         print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
        #         print("--------------------")
        #         print("..." + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100) : min(list_positions_together)]
        #             + '|||||'
        #             + row[0][min(list_positions_together):max(list_positions_together)]
        #             + '|||||'
        #             + row[0][max(list_positions_together) : (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)] + "...")
        #
        #         correct = input("\nIs this annotation correct? [y]es [n]o: ")
        #         if correct != 'n':
        #             annotation_string = '(' \
        #                                 + "".join(list_detected_negated_edges[idx][1].getCategory()[0]) \
        #                                 + ',False,' + row[0][list_positions[idx][1][0]:list_positions[idx][1][1]] \
        #                                 + ',' + row[0][list_positions[idx][0][0]:list_positions[idx][0][1]] \
        #                                 + ',' + str(min(list_positions_together)) \
        #                                 + ',' + str(max(list_positions_together)) \
        #                                 + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
        #                                 + ')'
        #             clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
        #             print('Added ' + annotation_string + ' to list of annotations.')
        #         print('\n')
        #
        ##################################################################################################

        # handle non case
        transcript_to_process = row[0]
        list_tokens = transcript_to_process.split()
        for i in range(len(list_tokens)):
            token = list_tokens[i]
            if token[0:3] == 'non':
                if len(token) == 3:
                    # non tender
                    potential_negated_concept = list_tokens[i+1]
                    not_string_to_test = "not " + potential_negated_concept + "."
                    list_non_edges = negations_pycontextnlp_individual_transcript(not_string_to_test)[0]
                    for idx in range(len(list_non_edges)):
                        if 'neg' in list_non_edges[idx][0].getCategory()[0]:

                            # get position of both "non" and the concept being negated
                            list_positions_together = []
                            non_sum = 0
                            for j in range(i):
                                non_sum += len(list_tokens[j]) + 1
                            list_positions_together.append(non_sum)
                            list_positions_together.append(non_sum + len(list_tokens[i]))
                            list_positions_together.append(non_sum + len(list_tokens[i]) + 1)
                            list_positions_together.append(non_sum + len(list_tokens[i]) + 1 + len(list_tokens[i+1]))

                            print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                            to_add = "".join(list_non_edges[idx][1].getCategory()[0].split('_'))
                            # set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

                            # print sentence being analyzed
                            print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
                            print("--------------------")
                            print("..."
                                  + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100): min(list_positions_together)]
                                  + '|||||'
                                  + row[0][min(list_positions_together):max(list_positions_together)]
                                  + '|||||'
                                  + row[0][max(list_positions_together): (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)]
                                  + "...")

                            correct = input("\nIs this annotation correct? [y]es [n]o: ")
                            if correct != 'n':
                                annotation_string = '(' \
                                                    + "".join(list_non_edges[idx][1].getCategory()[0]) \
                                                    + ',False,' + row[0][list_positions_together[2]:list_positions_together[3]] \
                                                    + ',' + row[0][list_positions_together[0]:list_positions_together[1]] \
                                                    + ',' + str(min(list_positions_together)) \
                                                    + ',' + str(max(list_positions_together)) \
                                                    + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
                                                    + ')'
                                clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
                                print('Added ' + annotation_string + ' to list of annotations.')
                            print('\n')

                elif token[3] == '-':
                    # non-tender
                    potential_negated_concept = token[4:]
                    not_string_to_test = "not " + potential_negated_concept + "."
                    list_non_edges = negations_pycontextnlp_individual_transcript(not_string_to_test)[0]
                    for idx in range(len(list_non_edges)):
                        if 'neg' in list_non_edges[idx][0].getCategory()[0]:

                            # get position of both "non" and the concept being negated
                            list_positions_together = []
                            non_sum = 0
                            for j in range(i):
                                non_sum += len(list_tokens[j]) + 1
                            list_positions_together.append(non_sum)
                            list_positions_together.append(non_sum + 3)
                            list_positions_together.append(non_sum + 4)
                            list_positions_together.append(non_sum + 4 + len(potential_negated_concept))

                            print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                            to_add = "".join(list_non_edges[idx][1].getCategory()[0].split('_'))
                            # set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

                            # print sentence being analyzed
                            print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
                            print("--------------------")
                            print("..."
                                  + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100): min(list_positions_together)]
                                  + '|||||'
                                  + row[0][min(list_positions_together):max(list_positions_together)]
                                  + '|||||'
                                  + row[0][max(list_positions_together): (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)]
                                  + "...")

                            correct = input("\nIs this annotation correct? [y]es [n]o: ")
                            if correct != 'n':
                                annotation_string = '(' \
                                                    + "".join(list_non_edges[idx][1].getCategory()[0]) \
                                                    + ',False,' + row[0][list_positions_together[2]:list_positions_together[3]] \
                                                    + ',' + row[0][list_positions_together[0]:list_positions_together[1]] \
                                                    + ',' + str(min(list_positions_together)) \
                                                    + ',' + str(max(list_positions_together)) \
                                                    + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
                                                    + ')'
                                clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
                                print('Added ' + annotation_string + ' to list of annotations.')
                            print('\n')

                else:
                    # nontender
                    potential_negated_concept = token[3:]
                    not_string_to_test = "not " + potential_negated_concept + "."
                    list_non_edges = negations_pycontextnlp_individual_transcript(not_string_to_test)[0]
                    for idx in range(len(list_non_edges)):
                        if 'neg' in list_non_edges[idx][0].getCategory()[0]:

                            # get position of both "non" and the concept being negated
                            list_positions_together = []
                            non_sum = 0
                            for j in range(i):
                                non_sum += len(list_tokens[j]) + 1
                            list_positions_together.append(non_sum)
                            list_positions_together.append(non_sum + 3)
                            list_positions_together.append(non_sum + 3)
                            list_positions_together.append(non_sum + 3 + len(potential_negated_concept))

                            print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                            to_add = "".join(list_non_edges[idx][1].getCategory()[0].split('_'))
                            # set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

                            # print sentence being analyzed
                            print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
                            print("--------------------")
                            print("..."
                                  + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100): min(list_positions_together)]
                                  + '|||||'
                                  + row[0][min(list_positions_together):max(list_positions_together)]
                                  + '|||||'
                                  + row[0][max(list_positions_together): (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)]
                                  + "...")

                            correct = input("\nIs this annotation correct? [y]es [n]o: ")
                            if correct != 'n':
                                annotation_string = '(' \
                                                    + "".join(list_non_edges[idx][1].getCategory()[0]) \
                                                    + ',False,' + row[0][list_positions_together[2]:list_positions_together[3]] \
                                                    + ',' + row[0][list_positions_together[0]:list_positions_together[1]] \
                                                    + ',' + str(min(list_positions_together)) \
                                                    + ',' + str(max(list_positions_together)) \
                                                    + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
                                                    + ')'
                                clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
                                print('Added ' + annotation_string + ' to list of annotations.')
                            print('\n')

        # handle dash negation case
        list_tokens = transcript_to_process.split()
        for i in range(len(list_tokens)):
            token = list_tokens[i]
            if token[0] == '-':
                if len(token) == 1:
                    # - DCAPBTLS
                    potential_negated_concept = list_tokens[i+1]
                    not_string_to_test = "not " + potential_negated_concept + "."
                    list_non_edges = negations_pycontextnlp_individual_transcript(not_string_to_test)[0]
                    for idx in range(len(list_non_edges)):
                        if 'neg' in list_non_edges[idx][0].getCategory()[0]:

                            # get position of both "non" and the concept being negated
                            list_positions_together = []
                            non_sum = 0
                            for j in range(i):
                                non_sum += len(list_tokens[j]) + 1
                            list_positions_together.append(non_sum)
                            list_positions_together.append(non_sum + len(list_tokens[i]))
                            list_positions_together.append(non_sum + len(list_tokens[i]) + 1)
                            list_positions_together.append(non_sum + len(list_tokens[i]) + 1 + len(list_tokens[i+1]))

                            print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                            to_add = "".join(list_non_edges[idx][1].getCategory()[0].split('_'))
                            # set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

                            # print sentence being analyzed
                            print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
                            print("--------------------")
                            print("..."
                                  + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100): min(list_positions_together)]
                                  + '|||||'
                                  + row[0][min(list_positions_together):max(list_positions_together)]
                                  + '|||||'
                                  + row[0][max(list_positions_together): (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)]
                                  + "...")

                            correct = input("\nIs this annotation correct? [y]es [n]o: ")
                            if correct != 'n':
                                annotation_string = '(' \
                                                    + "".join(list_non_edges[idx][1].getCategory()[0]) \
                                                    + ',False,' + row[0][list_positions_together[2]:list_positions_together[3]] \
                                                    + ',' + row[0][list_positions_together[0]:list_positions_together[1]] \
                                                    + ',' + str(min(list_positions_together)) \
                                                    + ',' + str(max(list_positions_together)) \
                                                    + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
                                                    + ')'
                                clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
                                print('Added ' + annotation_string + ' to list of annotations.')
                            print('\n')

                else:
                    # -DCAPBTLS
                    potential_negated_concept = token[1:]
                    not_string_to_test = "not " + potential_negated_concept + "."
                    list_non_edges = negations_pycontextnlp_individual_transcript(not_string_to_test)[0]
                    for idx in range(len(list_non_edges)):
                        if 'neg' in list_non_edges[idx][0].getCategory()[0]:

                            # get position of both "non" and the concept being negated
                            list_positions_together = []
                            non_sum = 0
                            for j in range(i):
                                non_sum += len(list_tokens[j]) + 1
                            list_positions_together.append(non_sum)
                            list_positions_together.append(non_sum + 1)
                            list_positions_together.append(non_sum + 1)
                            list_positions_together.append(non_sum + 1 + len(potential_negated_concept))

                            print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                            to_add = "".join(list_non_edges[idx][1].getCategory()[0].split('_'))
                            # set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

                            # print sentence being analyzed
                            print("..." + row[0][min(list_positions_together):max(list_positions_together)] + "...")
                            print("--------------------")
                            print("..."
                                  + row[0][(0 if min(list_positions_together) - 100 < 0 else min(list_positions_together) - 100): min(list_positions_together)]
                                  + '|||||'
                                  + row[0][min(list_positions_together):max(list_positions_together)]
                                  + '|||||'
                                  + row[0][max(list_positions_together): (len(row[0]) if max(list_positions_together) + 100 > len(row[0]) else max(list_positions_together) + 100)]
                                  + "...")

                            correct = input("\nIs this annotation correct? [y]es [n]o: ")
                            if correct != 'n':
                                annotation_string = '(' \
                                                    + "".join(list_non_edges[idx][1].getCategory()[0]) \
                                                    + ',False,' + row[0][list_positions_together[2]:list_positions_together[3]] \
                                                    + ',' + row[0][list_positions_together[0]:list_positions_together[1]] \
                                                    + ',' + str(min(list_positions_together)) \
                                                    + ',' + str(max(list_positions_together)) \
                                                    + ',' + row[0][min(list_positions_together):max(list_positions_together)] \
                                                    + ')'
                                clinical_text_df.iat[index, 1] = str(row[1]) + '\n' + annotation_string
                                print('Added ' + annotation_string + ' to list of annotations.')
                            print('\n')

    clinical_text_df.to_csv('data/exported_generated_annotations.csv')


def negations_pycontextnlp_individual_transcript(clinical_text):
    PYCONTEXTNLP_MODIFIERS = r'/' + os.getcwd() + '/data/pycontextnlp_modifiers.yml'
    PYCONTEXTNLP_TARGETS = r'/' + os.getcwd() + '/data/pycontextnlp_targets.yml'

    modifiers = itemData.get_items(PYCONTEXTNLP_MODIFIERS)
    targets = itemData.get_items(PYCONTEXTNLP_TARGETS)

    sentences = transcript_to_sentences_of_tokens(clinical_text, False)

    list_negated_edges = []
    list_positions = []
    curr_combined_length = 0

    for sentence in sentences:
        returned_negated_edges = pycontextnlp_markup_sentence(sentence.lower(), modifiers, targets)
        for edge in returned_negated_edges:
            list_positions.append(
                (
                    (curr_combined_length + edge[0].getSpan()[0], curr_combined_length + edge[0].getSpan()[1]),
                    (curr_combined_length + edge[1].getSpan()[0], curr_combined_length + edge[1].getSpan()[1])
                )
            )
        curr_combined_length += len(sentence) + 1
        list_negated_edges.extend(returned_negated_edges)

    return (list_negated_edges, list_positions)


def pycontextnlp_markup_sentence(s, modifiers, targets, prune_inactive=True):
    markup = pyConText.ConTextMarkup()

    markup.setRawText(s)
    markup.cleanText()

    markup.markItems(modifiers, mode="modifier")
    markup.markItems(targets, mode="target")

    markup.pruneMarks()
    markup.dropMarks('Exclusion')

    markup.applyModifiers()

    markup.pruneSelfModifyingRelationships()
    if prune_inactive:
        markup.dropInactiveModifiers()

    list_negated_edges = []

    for edge in markup.edges():
        # modifier_category = edge[0].getCategory()
        # if('neg' in modifier_category[0]):
        #     # print(edge)
        #     list_negated_edges.append(edge)
        list_negated_edges.append(edge)

    return list_negated_edges

def transcript_to_sentences_of_tokens(transcript_to_process, return_tokens=True):
    # first split the transcript by whitespace into a list
    list_tokens = transcript_to_process.split()

    # if a tokens last character is ':' or '-', add a special character ^ to the end of the token before it
    # indicating end of sentence.
    for i in range(len(list_tokens)):
        if (list_tokens[i][-1] == ':' or list_tokens[i][-1] == '-') \
                and (i != 0) \
                and (list_tokens[i - 1][-1] != '.') \
                and (list_tokens[i - 1][-1] != '!') \
                and (list_tokens[i - 1][-1] != '?'):
            list_tokens[i - 1] = list_tokens[i - 1] + '^'

    # a list of concatenations that do not signify end of sentence.
    list_concatenations = ['pt.', 'st.', 'dr.']

    # divide the tokens into their own individual lists by looking for punctuation characters
    # default punctuation characters are '.', '!', '?',
    list_sentences = []

    sentence = []
    while len(list_tokens) != 0:
        if (list_tokens[0][-1] == '.'
            or list_tokens[0][-1] == '!'
            or list_tokens[0][-1] == '?'
            or list_tokens[0][-1] == '^') \
                and list_tokens[0].lower() not in list_concatenations:

            # special character check, remove if it's there
            if list_tokens[0][-1] == '^':
                list_tokens[0] = list_tokens[0][:-1]

            sentence.append(list_tokens.pop(0))
            list_sentences.append(sentence)
            sentence = []
        else:
            sentence.append(list_tokens.pop(0))

    # if return_tokens == True, then return the sentences as lists of tokens
    # otherwise, return as individual strings
    if return_tokens:
        return list_sentences
    else:
        list_sentences_joined = []
        for sentence in list_sentences:
            list_sentences_joined.append(' '.join(sentence))
        return list_sentences_joined

def main():
    clinical_text_df = pd.read_excel("data/eimara_annotations.xls")

    # file used for testing purposes
    # clinical_text_df = pd.read_excel("data/test_opposite_concepts.xls")

    # pycontextnlp method
    negations_pycontextnlp(clinical_text_df)


if __name__ == "__main__":
    main()
