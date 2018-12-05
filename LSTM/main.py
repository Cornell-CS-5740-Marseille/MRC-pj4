import csv
import pickle

from LSTM.model import LSTM
from evaluate import calculate_precision, calculate_recall, calculate_f1
from prep import Preprocessing
from LSTM.params import Params as params
from keras.preprocessing.sequence import pad_sequences
import keras.models
import numpy as np
import matplotlib.pyplot as plt


def preprocessing(training_file):
    prep = Preprocessing(False)
    prep.load_file(training_file, False)
    passages = []
    questions = []
    answers = []
    answerable = []
    dictionary = prep.dictionary
    pickle_out = open("data/word_index_vocab.pickle", "wb")
    pickle.dump(prep.dictionary, pickle_out)
    for article in prep.articles:
        for paragraph in article.paragraphs:
            for question in paragraph.questions:
                possible = [int(question.possible)]
                passages.append(list(map(lambda x: dictionary[x], paragraph.words)))
                questions.append(list(map(lambda x: dictionary[x], question.words)))
                # if len(question.answers) == 0:
                #     answers.append([])
                # else:
                #     answers.append(list(map(lambda x: dictionary[x], question.answers[0].words)))
                answerable.append(possible)

    max_passage_len = max(list(map(lambda x: len(x), passages)))
    max_question_len = max(list(map(lambda x: len(x), questions)))
    params.MAX_SEQUENCE_LENGTH_PASSAGE = max_passage_len
    params.MAX_SEQUENCE_LENGTH_QUESTION = max_question_len
    # creating the number sequence of the context and question
    contexts = pad_sequences(passages, maxlen=params.MAX_SEQUENCE_LENGTH_PASSAGE)
    queries = pad_sequences(questions, maxlen=params.MAX_SEQUENCE_LENGTH_QUESTION)

    # creating the answer array encoding
    # answer_data = []
    # for items in answers:
    #     y = np.zeros(len(dictionary) + 1)
    #     for item in items:
    #         y[item] = 1
    #     answer_data.append(y)

    answers_ids = []
    answerable = np.array(answerable)

    return contexts, queries, answers_ids, answerable, dictionary


def train():
    contexts, queries, answers_ids, answerable, dictionary = preprocessing(params.training_file)
    model = LSTM(params.MAX_SEQUENCE_LENGTH_PASSAGE,
                 len(dictionary) + 1,
                 params.EMBED_HIDDEN_SIZE,
                 params.MAX_SEQUENCE_LENGTH_QUESTION,
                 params.RNN,
                 params.BATCH_SIZE,
                 params.EPOCHS,
                 contexts, queries, answers_ids, answerable)


def predict(test_file):
    prep = Preprocessing(False)
    prep.load_file(test_file, True)
    model = keras.models.load_model('data/LSTM_model.hdf5')
    pickle_out = pickle.load(open("data/word_index_vocab.pickle", "rb"))
    dictionary = {key: value for key, value in pickle_out.items()}

    passages = []
    questions = []

    for article in prep.articles:
        for paragraph in article.paragraphs:
            for question in paragraph.questions:
                passages.append(
                    list(map(lambda x: dictionary[x] if x in dictionary else len(dictionary), paragraph.words)))
                questions.append(
                    list(map(lambda x: dictionary[x] if x in dictionary else len(dictionary), question.words)))

    contexts = pad_sequences(passages, maxlen=params.MAX_SEQUENCE_LENGTH_PASSAGE)
    queries = pad_sequences(questions, maxlen=params.MAX_SEQUENCE_LENGTH_QUESTION)

    answerables = model.predict([contexts, queries])
    answerables = list(map(lambda x: 1 if x[0] > params.THRESHOLD else 0, answerables))

    with open('../output/output_LSTM.csv', 'w') as outfile:
        speech_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        speech_writer.writerow(['Id', 'Category'])
        index = 0
        for article in prep.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    speech_writer.writerow([question.id, int(answerables[index])])
                    index += 1


def evaluate(development_file):
    prep = Preprocessing(False)
    prep.load_file(development_file, False)
    model = keras.models.load_model('data/LSTM_model.hdf5')
    pickle_out = pickle.load(open("data/word_index_vocab.pickle", "rb"))
    dictionary = {key: value for key, value in pickle_out.items()}

    passages = []
    questions = []

    for article in prep.articles:
        for paragraph in article.paragraphs:
            for question in paragraph.questions:
                passages.append(
                    list(map(lambda x: dictionary[x] if x in dictionary else len(dictionary), paragraph.words)))
                questions.append(
                    list(map(lambda x: dictionary[x] if x in dictionary else len(dictionary), question.words)))

    contexts = pad_sequences(passages, maxlen=params.MAX_SEQUENCE_LENGTH_PASSAGE)
    queries = pad_sequences(questions, maxlen=params.MAX_SEQUENCE_LENGTH_QUESTION)

    answerables = model.predict([contexts, queries])
    threshold = 0
    times = 5
    upper = 1
    accuracies = []
    f1s = []

    for x in range(times + 1):
        true_negatives = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        index = 0
        temp_answerables = list(map(lambda x: 1 if x[0] > threshold else 0, answerables))

        for article in prep.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    answerable = int(temp_answerables[index])
                    if answerable == 0 and question.possible == 0:
                        true_negatives += 1
                    elif answerable == 1 and question.possible == 1:
                        true_positives += 1
                    elif answerable == 0 and question.possible == 1:
                        false_positives += 1
                    elif answerable == 1 and question.possible == 0:
                        false_negatives += 1
                    index += 1

        precision = calculate_precision(true_positives, false_positives)
        accuracy = float(true_negatives + true_positives) / len(temp_answerables)
        recall = calculate_recall(true_positives, false_negatives)
        f_1 = calculate_f1(precision, recall)
        accuracies.append(accuracy)
        f1s.append(f_1)
        print("{0:.2f}".format(round(accuracy * 100, 2)) + "\%", "&",
              "{0:.2f}".format(round(precision * 100, 2)) + "\%", "&",
              "{0:.2f}".format(round(recall * 100, 2)) + "\%", "&",
              "{0:.2f}".format(round(f_1 * 100, 2)) + "\%", "&",
              "threshold:", threshold)
        threshold += upper / float(times)
    high_accuracy = max(accuracies)
    high_f1 = max(f1s)
    print("Max accuracy:", high_accuracy, accuracies.index(high_accuracy))
    print("Max f-1 score:", high_f1, f1s.index(high_f1))

def loadHistory():
    history = pickle.load(open("data/trainHistory.pickle", "rb"))

    # summarize history for loss
    plt.plot(history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("data/loss_LSTM.png")

# train()
predict(params.test_file)
# evaluate(params.development_file)
# loadHistory()