import pyConTextNLP.pyConText as pyConText
import pyConTextNLP.itemData as itemData
import pandas as pd
import os


def negations_pycontextnlp(clinical_text_df):
    total_neg_concepts_detected = 0
    total_expected_negated_concepts = 0

    precision_sum = 0.
    recall_sum = 0.
    f1_sum = 0.
    total_transcripts_passed = 0.

    for index, row in clinical_text_df.iterrows():
        # if the row has a NaN value in the transcripts column, skip it
        if (pd.isna(clinical_text_df.iloc[index, 0])):
            continue

        print("Transcript " + str(index) + ", row " + str(index + 2) + ":")

        # modify the transcript by replacing all whitespace with individual spaces
        transcript_to_process = ' '.join(transcript_to_sentences_of_tokens(row[0], False))
        clinical_text_df.iat[index, 0] = transcript_to_process

        # print("Detected negated edges:")
        list_detected_negated_edges, list_positions = negations_pycontextnlp_individual_transcript(
            transcript_to_process)

        print("Detected negated concepts:\n")
        set_detected_negated_concepts = set()
        for idx in range(len(list_detected_negated_edges)):

            # handle opposite case
            if 'opposite' in list_detected_negated_edges[idx][1].getCategory()[0]:

                list_positions_together = []
                for i in range(2):
                    for j in range(2):
                        list_positions_together.append(list_positions[idx][i][j])

                # print sentence being analyzed
                if max(list_positions_together) - min(list_positions_together) < 105:
                    print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                to_add = "".join(list_detected_negated_edges[idx][1].getCategory()[0].split('_opposite'))
                set_detected_negated_concepts.add(to_add)
                print("negated concept '" + to_add + "' detected at position ("
                      + str(list_positions[idx][0][0]) + ", " + str(list_positions[idx][0][1])
                      + ") (" + transcript_to_process[list_positions[idx][0][0]:list_positions[idx][0][1]] + "), ("
                      + str(list_positions[idx][1][0]) + ", " + str(list_positions[idx][1][1]) + ") ("
                      + transcript_to_process[list_positions[idx][1][0]:list_positions[idx][1][1]] + ")\n")

            # handle negative edge case
            elif 'neg' in list_detected_negated_edges[idx][0].getCategory()[0]:

                list_positions_together = []
                for i in range(2):
                    for j in range(2):
                        list_positions_together.append(list_positions[idx][i][j])

                # print sentence being analyzed
                if max(list_positions_together) - min(list_positions_together) < 105:
                    print("..." + transcript_to_process[min(list_positions_together):max(list_positions_together)] + "...")

                to_add = "".join(list_detected_negated_edges[idx][1].getCategory()[0].split('_'))

                # ignore detections longer than a certain number of characters in length
                if max(list_positions_together) - min(list_positions_together) < 105:
                    set_detected_negated_concepts.add(to_add)

                    print("negated concept '" + to_add + "' detected at position ("
                          + str(list_positions[idx][0][0]) + ", " + str(list_positions[idx][0][1]) + ") ("
                          + transcript_to_process[list_positions[idx][0][0]:list_positions[idx][0][1]] + "), ("
                          + str(list_positions[idx][1][0]) + ", " + str(list_positions[idx][1][1]) + ") ("
                          + transcript_to_process[list_positions[idx][1][0]:list_positions[idx][1][1]] + ")\n")

        # handle non case
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
                            set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

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
                            set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

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
                            set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

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
                            set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

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
                            set_detected_negated_concepts.add(to_add)

                            print("negated concept '" + to_add + "' detected at position ("
                                + str(list_positions_together[0]) + ", " + str(list_positions_together[1]) + ") ("
                                + transcript_to_process[list_positions_together[0]:list_positions_together[1]] + "), ("
                                + str(list_positions_together[2]) + ", " + str(list_positions_together[3]) + ") ("
                                + transcript_to_process[list_positions_together[2]:list_positions_together[3]] + ")\n")

        print(set_detected_negated_concepts)

        print("Expected negated concepts:")
        if pd.isnull(row[1]):
            expected_negated_concepts = []
        else:
            expected_concepts = "".join(row[1].split())
            expected_concepts = expected_concepts[1:-1].split(')(')
            expected_negated_concepts = set()
            for concept in expected_concepts:
                if ('false' in concept or 'False' in concept):
                    # TEMPORARY: ignore corner cases of related concepts
                    # if(
                    #     'breath' not in concept
                    #     and 'shortnessofbreath' not in concept
                    #
                    #     and 'lightheadedness' not in concept
                    #     and 'dizziness' not in concept
                    #
                    #     and 'hyperthermia' not in concept
                    #     and 'fever' not in concept
                    # ):

                    # replace underscores with empty string because i'm dumb
                    expected_negated_concepts.add(concept.split(',')[0].replace('_', ''))
        print(expected_negated_concepts)

        # comment this out to include transcript that have no negated concepts
        ######################################################################
        if len(expected_negated_concepts) == 0:
            print('\n\n')
            continue
        ######################################################################

        true_positives = 0.
        false_positives = 0.
        for concept in set_detected_negated_concepts:
            if concept in expected_negated_concepts:
                true_positives += 1.
            else:
                false_positives += 1.
        false_negatives = len(expected_negated_concepts) - true_positives

        if true_positives == 0 and false_positives == 0 and false_negatives == 0:
            transcript_precision = 1.
            transcript_recall = 1.
            transcript_f1 = 1.
        elif true_positives == 0 and (false_positives > 0 or false_negatives > 0):
            transcript_precision = 0.
            transcript_recall = 0.
            transcript_f1 = 0.
        else:
            transcript_precision = true_positives / (true_positives + false_positives)
            transcript_recall = true_positives / (true_positives + false_negatives)
            transcript_f1 = 2 * (
                    (transcript_precision * transcript_recall) / (transcript_precision + transcript_recall))

        print("\nPrecision for this transcript: " + str(transcript_precision))
        print("Recall for this transcript: " + str(transcript_recall))
        print("F1 for this transcript: " + str(transcript_f1))

        total_neg_concepts_detected += true_positives
        total_expected_negated_concepts += len(expected_negated_concepts)
        precision_sum += transcript_precision
        recall_sum += transcript_recall
        f1_sum += transcript_f1
        total_transcripts_passed += 1.

        print('\n\n')

    print('###################################################')
    print('Total number of negated concepts detected / All negated concepts: ' + str(
        total_neg_concepts_detected) + '/' + str(total_expected_negated_concepts))
    print('Average precision: ' + str(precision_sum / total_transcripts_passed))
    print('Average recall: ' + str(recall_sum / total_transcripts_passed))
    print('Average F1: ' + str(f1_sum / total_transcripts_passed))


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
        # add 1 to account for stripped space
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
