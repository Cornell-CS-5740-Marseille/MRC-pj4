class Params(object):
    article_len = 10
    sentence_len = 1

    l2_lambda = 0.0001
    vocab_size = 10
    embed_size = 300  # 128
    ner_vocab_sz = 10
    pos_vocab_sz = 10
    learning_rate = 0.0001
    training_file = "../data/training.json"