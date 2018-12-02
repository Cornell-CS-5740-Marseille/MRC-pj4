import tensorflow as tf
from MEMEN.model import Memen
from MEMEN.word2vec import word2vec
from prep import Preprocessing
from MEMEN.params import Params as param
import numpy as np

def train():
    ckpt_dir='models/'
    #.create model
    preprocessing = Preprocessing(True)
    preprocessing.load_file(param.training_file)
    param.vocab_size = preprocessing.dict_index
    model = Memen(param.l2_lambda, param.context_len, param.sentence_len, param.vocab_size,
                  param.embed_size, param.hidden_size, param.learning_rate)
    word_embedding = word2vec(param.word2vec)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        index = 0
        for article in preprocessing.articles:
            for paragraph in article.paragraphs:
                # paragraph.words = ["we", "are", "fake"]

                # context_embedding = word_embedding.word_embedding(paragraph.words)

                context = np.reshape(np.array(preprocessing.word_ids(paragraph.words)), [1, len(paragraph.words)])
                for question in paragraph.questions:
                    # generate data
                    answerable = [question.possible]
                    # question_embedding = word_embedding.word_embedding(question.words)
                    query = np.reshape(np.array(preprocessing.word_ids(question.words)), [1, len(question.words)])
                    start_point = np.reshape(np.array(list(map(lambda x: x.answer_start, question.answers))), [1])
                    end_point = np.reshape(np.array(list(map(lambda x: x.answer_start + len(x.text), question.answers))), [1])

                    # feed dict
                    fetch_dict = [model.loss_val, model.accurate_start, model.accurate_end, model.predictions_start,
                                  model.predictions_end, model.train_op]

                    feed_dict = {model.context: context, model.query: query, model.label_start: start_point,
                                 model.label_end: end_point}
                    # run with session
                    loss, accuracy_start, accuracy_end, predictions_start, predictions_end, _ \
                        = sess.run(fetch_dict, feed_dict=feed_dict)
                    # print result and status
                    print(index, "paragraph:", context, "; query:", query, "; start_point:", start_point, "; end_point:",
                          end_point)
                    print(index, "; loss:", loss, "; accuracy_start:", accuracy_start, "; accuracy_end:", accuracy_end,
                          "; predictions_start:", predictions_start, "; predictions_end:", predictions_end)
                    # save model
                    if index % 300 == 0:
                        save_path = ckpt_dir + "model.ckpt"
                        saver.save(sess, save_path, global_step=index)
def predict():
    #1.generate data
    preprocessing = Preprocessing(True)
    preprocessing.load_file(param.training_file)
    #2.create model

    #3.feed data

    #4.predict use model in the checkpoint
    pass

train()