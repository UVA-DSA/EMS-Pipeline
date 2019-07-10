#Elizabeth Shelton ejs6ar
"""
Processes text files of interventions truth/output into lists and calculates the recall, precision, and f1 score.
Format for txt files:
1) ground truth file: copy straight from an Excel sheet the column of interventions typed simply from the RAA data
2) pipeline results file: copy and paste each test result from the Pipeline to an Excel sheet; copy and paste the column into a text file
"""

def get_results_interventions(path):
    """
    Converts the pipeline output into a master list of interventions per case
    :param path: the location of the .txt file with the pipeline output
    :return: results: list of lists; each element of the overall list includes all the interventions for single case
            all_r: a set of all interventions in all cases (for a pooled calculation)
    """
    results = []
    all = set()
    file = open(path)
    for group in file:
        if len(group) < 5:
            continue
        # print(group)
        group = group.strip("\"' \n()")
        # print(group)
        tuple_list = group.split("), (")
        for i in range(len(tuple_list)):
            tuple_list[i] = tuple_list[i].strip("'.1234567890\n() ,").lower()
            # print(tuple_list[i])
            all.add(tuple_list[i])
        results.append(tuple_list)
    # hold = []
    # for j in results.keys():
    #     if results[j] == ['']:
    #         hold.append(j)
    # for m in hold:
    #     results.pop(m)
    file.close()
    return results, all

def get_truth_interventions(path):
    """
    Converts the truth, in a text file, into a master list of interventions per case
    :param path: the location of the .txt file with the ground truth
    :return: results: list of lists; each element of the overall list includes all the interventions for single case
            all_r: a set of all interventions in all cases (for a pooled calculation)
    """
    truth = []
    all = set()
    file = open(path)
    for group in file:
        group = group.strip().split(",")
        for i in range(len(group)):
            item = group[i]
            item = item.strip(" ")
            item = item.lower()
            if item == "hospital contact":
                item = "transport"
            elif item == "IV":
                item = "normal saline"
            elif item == "Restraints":
                item = "physical restraint"
            elif "narcan" in item:
                item = "narcan"
            elif "ondansetron" in item:
                item = "ondansetron"
            group[i] = item
            all.add(item)
        truth.append(group)
    file.close()
    return truth, all


def get_tp(truth, results):
    """
    :param truth: a single list of ground truth interventions
    :param results: a single list of pipeline-returned interventions
    :return: a list of true positives (in the ground truth, returned by pipeline)
    """
    # print("Truth: " + str(truth) + "\t Results: " + str(results))
    tP = []
    for i in results:
        if i in truth:
            tP.append(i)
    # print("TP: " + str(tP))
    return tP

def get_fp(truth, results):
    """
    :param truth: a single list of ground truth interventions
    :param results: a single list of pipeline-returned interventions
    :return: a list of false positives (not in the ground truth, returned by pipeline)
    """
    fP = []
    for i in results:
        if i not in truth:
            fP.append(i)
    # print("FP: " + str(fP))
    return fP

def get_fn(truth, results):
    """
    :param truth: a single list of ground truth interventions
    :param results: a single list of pipeline-returned interventions
    :return: a list of false negatives (in the ground truth, not returned by pipeline)
    """
    fN = []
    for i in truth:
        if i not in results:
            fN.append(i)
    # print("FN: " + str(fN))
    return fN

def calc_precision(truth, results):
    tP = get_tp(truth, results)
    fP = get_fp(truth, results)
    return len(tP)/(len(tP)+len(fP))

def calc_recall(truth, results):
    tP = get_tp(truth, results)
    fN = get_fn(truth, results)
    return len(tP)/(len(tP)+len(fN))

def calc_f1(truth, results):
    prec = calc_precision(truth, results)
    rec = calc_recall(truth, results)
    denom = (prec + rec)
    if denom == 0:
        return 0
    return 2*((prec * rec)/denom)


def main():
    path_results = "C:/Users/Student/OneDrive/Documents/Summer 2019 Research/Week 2-Testing/interv.txt"
    path_truth = "C:/Users/Student/OneDrive/Documents/Summer 2019 Research/Week 2-Testing/ground.txt"

    # Get results
    results, all_r = get_results_interventions(path_results)
    # print(results)
    # print(len(results))

    # Get truth
    truth, all_t = get_truth_interventions(path_truth)
    # print(truth)
    # print(len(truth))


    # Calculating metrics for each case
    # WITH LABELS
    # for i in range(12):
    #     print("Case " + str(i) + ":\t")
    #     print("Precision: " + str(calc_precision(truth[i], results[i])))
    #     print("Recall: " + str(calc_recall(truth[i], results[i])))
    #     # print("F1: " + str(calc_f1(truth[i], results[i])))
    #     print()

    # JUST NUMBERS
    # for i in range(12):
        # print(calc_precision(truth[i], results[i]))
        # print(calc_recall(truth[i], results[i]))
        # print(calc_f1(truth[i], results[i]))
        # print()

    # Metrics for pooled data
    # print(calc_precision(all_t, all_r))
    # print(calc_recall(all_t, all_r))
    # print(calc_f1(all_t, all_r))

    # print("Precision: " + str(calc_precision(truth[1], results[1])))
    # print("Recall: " + str(calc_recall(truth[1], results[1])))
