class Params():
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 100
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 100

    MAX_SEQUENCE_LENGTH_PASSAGE = 550
    MAX_SEQUENCE_LENGTH_QUESTION = 25
    MAX_SEQUENCE_LENGTH_ANSWER = 20
    MAX_NB_WORDS = 90000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2

    training_file = "../data/training.json"
    test_file = "../data/training.json"