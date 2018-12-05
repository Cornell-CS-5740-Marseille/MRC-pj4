from keras.layers import recurrent
class Params():
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 100
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 64
    EPOCHS = 120

    MAX_SEQUENCE_LENGTH_PASSAGE = 653
    MAX_SEQUENCE_LENGTH_QUESTION = 40
    MAX_SEQUENCE_LENGTH_ANSWER = 20
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    THRESHOLD = 0.62

    training_file = "../data/training.json"
    test_file = "../data/testing.json"
    development_file = "../data/development.json"
