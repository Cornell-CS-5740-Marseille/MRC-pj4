import tensorflow as tf
from MEMEN.model import Memen
from prep import Preprocessing
from MEMEN.params import Params as param


def train():
    ckpt_dir='models/'
    #.create model
    model=Memen(param.l2_lambda, param.article_len, param.sentence_len, param.vocab_size, param.embed_size,
                param.ner_vocab_sz, param.pos_vocab_sz, param.learning_rate)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        preprocessing = Preprocessing(True)
        preprocessing.load_file(param.training_file)
        index = 0
        for article in preprocessing.articles:
            for paragraph in article.paragraphs:
                context = [paragraph.context]
                for question in paragraph.questions:
                    answerable = question.possible
                    question_words = [question.words]

                    # feed dict
                    fetch_dict = [model.loss_val, model.accurate_start, model.accurate_end, model.predictions_start,
                                  model.predictions_end, model.train_op]
                    feed_dict = {model.context: paragraph, model.query: query, model.label_start: start_point,
                                 model.label_end: end_point}
                    # run with session
                    loss, accuracy_start, accuracy_end, predictions_start, predictions_end, _ = sess.run(fetch_dict,
                                                                                                         feed_dict=feed_dict)
                    # print result and status
                    print(index, "paragraph:", paragraph, ";query:", query, ";start_point:", start_point, ";end_point:",
                          end_point)
                    print(index, ";loss:", loss, ";accuracy_start:", accuracy_start, ";accuracy_end:", accuracy_end,
                          ";predictions_start:", predictions_start, ";predictions_end:", predictions_end)
                    # save model
                    if i % 300 == 0:
                        save_path = ckpt_dir + "model.ckpt"
                        saver.save(sess, save_path, global_step=i)
def predict():
    #1.generate data

    #2.create model

    #3.feed data

    #4.predict use model in the checkpoint
    pass

train()