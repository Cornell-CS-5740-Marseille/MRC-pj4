import json
import os
import sys


def evaluate(preds, test_set):
    """

    :param preds: A dictionary mapping question IDs to:
                    -> 0 if the question does not have a plausible answer
                    -> 1 if the question does have a plausible answer

    :param test_set: A dictionary of the following form:

    test_set["data"] -> list of titles
        title["title"] -> title name
        title["paragraphs"] -> list of paragraphs
            paragraph["context"] -> data which might answer questions in this paragraph
            paragraph["qas"] -> list of questions
                question["question"] -> question content
                question["id"] -> question ID

    :return: A 3-tuple consisting of (precision, recall, f1_score)
    """

    correct_mapping = {}

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for title in test_set["data"]:
        for paragraph in title["paragraphs"]:
            for question in paragraph["qas"]:
                if question["is_impossible"]:
                    correct_mapping[question["id"]] = 0
                else:
                    correct_mapping[question["id"]] = 1

    for id, pred in correct_mapping.items():
        if pred == 0 and preds[id] == 0:
            true_negatives += 1
        elif pred == 1 and preds[id] == 1:
            true_positives += 1
        elif pred == 0 and preds[id] == 1:
            false_positives += 1
        elif pred == 1 and preds[id] == 0:
            false_negatives += 1

    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f_1 = calculate_f1(precision, recall)
    return precision, recall, f_1


def calculate_precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)


def calculate_recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)


def calculate_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall)


if __name__ == "__main__":

    # collect args

    try:
        path_to_data = sys.argv[1]
    except:
        path_to_data = os.path.join(os.getcwd(), "development.json")

    try:
        path_to_predictions = sys.argv[2]
    except:
        path_to_predictions = os.path.join(os.getcwd(), "sample.json")

    # determine paths/names
    current_loc = os.getcwd()
    data_path, data_file_name = path_to_data.rsplit(os.sep, 1)
    predictions_path, predictions_file_name = path_to_predictions.rsplit(os.sep, 1)

    # fetch predictions
    predictions = {}
    os.chdir(predictions_path)
    with open(predictions_file_name, "r") as fileHandle:
        preds = json.load(fileHandle)
    os.chdir(current_loc)

    # fetch test set
    os.chdir(data_path)
    with open(data_file_name, "r") as fileHandle:
        test_set = json.load(fileHandle)
    os.chdir(current_loc)

    # determine performance
    p, r, f = evaluate(preds, test_set)
    print("precision score: " + str(p))
    print("recall score: " + str(r))
    print("f_1 measure: " + str(f))
