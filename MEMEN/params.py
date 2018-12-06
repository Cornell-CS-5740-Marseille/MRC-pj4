class Params(object):
    context_len = 109
    sentence_len = 11

    l2_lambda = 0.0001
    vocab_size = 20
    hidden_size = 300
    embed_size = 300  # 128
    learning_rate = 0.0001

    word2vec = "./models/GoogleNews-vectors-negative300.bin"
    training_file = "../data/training.json"
    test_file = "../data/training.json"