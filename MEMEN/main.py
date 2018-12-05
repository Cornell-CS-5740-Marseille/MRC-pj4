import csv
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from MEMEN.model import Memen
from MEMEN.simple_model import MemNN
from evaluate import calculate_precision, calculate_recall, calculate_f1
from prep import Preprocessing
from MEMEN.params import Params as param
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import keras.models


def train():
    #.create model
    preprocessing = Preprocessing(False)
    preprocessing.load_file(param.training_file, False)
    param.vocab_size = preprocessing.dict_index + 1
    model = Memen(param.l2_lambda, param.context_len, param.sentence_len, param.vocab_size,
                  param.embed_size, param.hidden_size, param.learning_rate)
    pickle_out = open("data/word_index_vocab.pickle", "wb")
    pickle.dump(preprocessing.dictionary, pickle_out)
    end_index = len(preprocessing.dictionary)
    with open("Output.txt", "w") as text_file:
        runTraining(preprocessing, model, text_file)


def runTraining(preprocessing, model, text_file):
    ckpt_dir = 'models/'
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        index = 0
        for article in preprocessing.articles:
            for paragraph in article.paragraphs:
                # paragraph.words = ["we", "are", "fake"]

                # context_embedding = word_embedding.word_embedding(paragraph.words)
                word_ids = preprocessing.word_ids(paragraph.words)
                for question in paragraph.questions:
                    query_ids = preprocessing.word_ids(question.words)
                    start = 0
                    end = 0
                    for answer in question.answers:
                        start = answer.answer_start
                        end = start + len(answer.text)
                    context, query, start_point, end_point, possible = \
                        generateData(word_ids, query_ids, start, end, question.possible)

                    # feed dict
                    fetch_dict = [model.loss_val, model.accurate_answerable, model.predictions_answerable,
                                  model.train_op]

                    feed_dict = {model.context: context, model.query: query, model.answerable: possible}
                    # run with session
                    loss, accurate_answerable, predictions_answerable, _ \
                        = sess.run(fetch_dict, feed_dict=feed_dict)
                    # print result and status
                    text_file.write(str(loss))
                    print(index, "; loss:", loss, "; accurate_answerable:", accurate_answerable,
                          "; predictions_answerable:", predictions_answerable)
                    index += 1
                    # save model
                    if index % 300 == 0:
                        save_path = ckpt_dir + "model.ckpt"
                        saver.save(sess, save_path)
                    if index == 3600:

                        return

def train_simple():
    contexts, queries, answers_ids, answerable, dictionary = preprocessing(param.training_file)
    model = MemNN(param.context_len,
                 len(dictionary) + 1,
                 param.embed_size,
                 param.sentence_len,
                 param.batch_size,
                 param.epoch_size,
                 contexts, queries, answers_ids, answerable)

def generateData(context, question, start, end, answerable):
    paragraph = np.reshape([context[x] if x < len(context) else 209690 for x in range(param.context_len)], [1, param.context_len])
    query = np.reshape([question[x] if x < len(question) else 209690 for x in range(param.sentence_len)], [1, param.sentence_len])
    start_point = np.reshape([start], [1])
    end_point = np.reshape([end], [1])
    possible = np.reshape([answerable], [1])
    return paragraph, query, start_point, end_point, possible

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
    param.context_len = max_passage_len
    param.sentence_len = max_question_len
    # creating the number sequence of the context and question
    contexts = pad_sequences(passages, maxlen=param.context_len)
    queries = pad_sequences(questions, maxlen=param.sentence_len)


    answers_ids = []
    answerable = np.array(answerable)

    return contexts, queries, answers_ids, answerable, dictionary

def predict():
    #1.generate data
    preprocessing = Preprocessing(False)
    preprocessing.load_file(param.training_file, True)
    #2.create model
    end_index = 209690
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models/model.ckpt.meta')
        # Restore variables from disk.
        saver.restore(sess, "models/model.ckpt")
        print("Model restored.")
        index = 0
        for article in preprocessing.articles:
            for paragraph in article.paragraphs:
                word_ids = preprocessing.word_ids(paragraph.words)
                passage_words = [word_ids[x] if x < len(word_ids) else end_index for x in range(param.context_len)]
                context = np.reshape(np.array(passage_words), [1, param.context_len])
                for question in paragraph.questions:
                    query_ids = preprocessing.word_ids(question.words)
                    query_words = [query_ids[x] if x < len(query_ids) else end_index for x in range(param.sentence_len)]
                    query = np.reshape(np.array(query_words), [1, param.sentence_len])
                    graph = tf.get_default_graph()
                    model_context = graph.get_tensor_by_name("context:0")
                    model_query = graph.get_tensor_by_name("query:0")



                    #3.feed data
                    # feed dict
                    op_to_restore = graph.get_tensor_by_name("prediction_answerable:0")

                    feed_dict = {model_context: context, model_query: query}
                    #4.predict use model in the checkpoint
                    predictions_answerable= sess.run(op_to_restore, feed_dict)

                    print(index, "predictions_answerable:", predictions_answerable)
                    index += 1

def simple_predict(test_file):
    prep = Preprocessing(False)
    prep.load_file(test_file, True)
    model = keras.models.load_model('data/MEMNN_model.hdf5')
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

    contexts = pad_sequences(passages, maxlen=param.context_len)
    queries = pad_sequences(questions, maxlen=param.sentence_len)

    answerables = model.predict([contexts, queries])
    answerables = list(map(lambda x: 1 if x[0] > param.THRESHOLD else 0, answerables))

    with open('../output/output_MEMNN.csv', 'w') as outfile:
        speech_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        speech_writer.writerow(['Id', 'Category'])
        index = 0
        for article in prep.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    speech_writer.writerow([question.id, int(answerables[index])])
                    index += 1

def loadHistory():
    history = pickle.load(open("data/trainHistory.pickle", "rb"))

    # summarize history for loss
    plt.plot(history['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig("data/loss_MEMNN.png")

def evaluate(development_file):
    prep = Preprocessing(False)
    prep.load_file(development_file, False)
    model = keras.models.load_model('data/MEMNN_model.hdf5')
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

    contexts = pad_sequences(passages, maxlen=param.context_len)
    queries = pad_sequences(questions, maxlen=param.sentence_len)

    answerables = model.predict([contexts, queries])
    threshold = 0
    times = 100
    upper = 1
    accuracies = []
    f1s = []

    for x in range(times):
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


# train()
# predict()
# simple_predict(param.test_file)
train_simple()
# loadHistory()
# evaluate(param.development_file)