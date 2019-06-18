#Elizabeth Shelton ejs6ar
"""
Processes text files of concepts truth/output into lists and calculates the recall, precision, and f1 score.
Format for txt files:
1) ground truth file: copy straight from an Excel sheet the column of concepts in the following format for each cell:
-four values per concept, comma-separated: concept, True/False (negation), value, text wording
-each concept separated by a semicolon ; and if possible (I don't think required) a newline
2) pipeline results file: copy and paste each test result from the Pipeline to an Excel sheet; copy and paste the column into a text file
"""
from interventions import get_tp, get_fp, get_fn, calc_recall, calc_precision, calc_f1


def get_truth_concepts(path):
    """
    Converts the truth, in a text file, into a master list of concepts per case
    :param path: the location of the .txt file with the pipeline output
    :return: truth: a list of lists of lists. Master list: contains all data. Each element of master list: a list of concepts for each case. Each case element: a concept
    """
    file = open(path)
    truth = []
    all_text = file.read()
    file.close()
    draft_list = all_text.strip().split('"')
    for item in draft_list:
        if item != '':
            case_list = item.strip().split(";")
            for i in range(len(case_list)):
                tup = case_list[i].strip()
                tup = tup.split(",")
                for j in range(len(tup)):
                    tup[j] = tup[j].strip()
                case_list[i] = tup
            for thing in case_list:
                if thing == ['']:
                    case_list.remove(thing)
            if len(case_list):
                # print(case_list)
                truth.append(case_list)
    return truth

def get_results_concepts(path):
    """
    Converts the pipeline output into a master list of concepts per case
    :param path: the location of the .txt file with the pipeline output
    :return: results: a list of lists of lists. Master list: contains all data. Each element of master list: a list of concepts for each case. Each case element: a concept
    """
    file = open(path)
    all_text = file.read()
    file.close()
    results = []
    draft_list = all_text.strip().split('"')
    for item in draft_list:
        if item != '':
            case_list = item.strip().split(")\n(")
            # print(case_list)
            for i in range(len(case_list)):
                tup = case_list[i].strip().strip("()")
                # print(tup)
                tup = tup.split(",")
                for j in range(len(tup)):
                    tup[j] = tup[j].strip("' \"")
                case_list[i] = tup
            for thing in case_list:
                if thing == ['']:
                    case_list.remove(thing)
            if len(case_list):
                # print(case_list)
                results.append(case_list)
    return results



def process_with_text(concepts):
    """
    The strictest way to compare concepts; looks at concept, negation, value, and text signal
    :param concepts: the truth or results master list
    :return: concepts: the master list, but each concept pared down to only [concept, T/F, value, text], and converted to string
    """
    for i in range(len(concepts)):
        for j in range(len(concepts[i])):
            concepts[i][j] = str(concepts[i][j][:4])
    return concepts



def process_no_text(concepts):
    """
    An intermediate-strictness way to compare concepts; compares the concept, negation, and value but not (for numerical values) the surrounding text
    :param concepts: the truth or results master list
    :return: concepts: the master list, but each concept pared down to only [concept, T/F, value] and converted to string
    """
    for i in range(len(concepts)):
        for j in range(len(concepts[i])):
            concepts[i][j] = str(concepts[i][j][:3])
    return concepts

def process_tf(concepts):
    """
    The least strict way to compare concepts; compares only the concept and its negation; useful for whether or not the concept and negation is correct and disregards differences in context
    :param concepts:
    :return: concepts: the master list, but each concept pared down to only [concept, T/F] and converted to string
    """
    for i in range(len(concepts)):
        for j in range(len(concepts[i])):
            concepts[i][j] = str(concepts[i][j][:2])
    return concepts


def main2():
    path_results = "C:/Users/Student/OneDrive/Documents/Summer 2019 Research/Week 3-Testing/extracted.txt"
    path_truth = "C:/Users/Student/OneDrive/Documents/Summer 2019 Research/Week 3-Testing/concepts.txt"

    # Get results
    results = get_results_concepts(path_results)
    # print(len(results))


    # Process results
    results = process_with_text(results)
    # results = process_no_text(results)
    # results = process_tf(results)

    # print(process_with_text(results)[0])
    # print(process_no_text(results)[0])
    # print(process_tf(results)[0])]

    # results[11].sort()
    # print(results[11])


    # Get truth
    truth = get_truth_concepts(path_truth)
    # print(len(truth))

    # Process truth
    truth = process_with_text(truth)
    # truth = process_no_text(truth)
    # truth = process_tf(truth)

    # print(process_with_text(truth)[0])
    # print(process_no_text(truth)[0])
    # print(process_tf(truth)[0])

    # truth[11].sort()
    # print(truth[11])


    # Calculating the metrics for each case
    # WITH LABELS
    for i in range(12):
        print("Case " + str(i) + ":\t")
        print("Precision: " + str(calc_precision(truth[i], results[i])))
        print("Recall: " + str(calc_recall(truth[i], results[i])))
        print("F1: " + str(calc_f1(truth[i], results[i])))
        print()

    # JUST NUMBERS
    # for i in range(12):
        # print(calc_precision(truth[i], results[i]))
        # print(calc_recall(truth[i], results[i]))
        # print(calc_f1(truth[i], results[i]))
        # print()


    # print("Precision: " + str(calc_precision(truth[1], results[1])))
    # print("Recall: " + str(calc_recall(truth[1], results[1])))


main2()