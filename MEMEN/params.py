class Params(object):
    context_len = 653
    sentence_len = 40

    l2_lambda = 0.0001
    vocab_size = 20
    hidden_size = 32
    embed_size = 100  # 128
    learning_rate = 0.0001
    batch_size = 32
    epoch_size = 50

    THRESHOLD = 0.9

    word2vec = "./models/GoogleNews-vectors-negative300.bin"
    training_file = "../data/training.json"
    test_file = "../data/test.json"
    development_file = "../data/development.json"