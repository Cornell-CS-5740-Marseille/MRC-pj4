import pickle

from keras import layers
from keras.models import Model

class LSTM():
    def __init__(self, context_len, vocab_size, embed_size, query_len, model, batch_size, epoch_size,
                 data_context, data_query, data_answer, answerable):
        # encoding the passage
        context = layers.Input(shape=(context_len,), dtype='int32')
        context_encoding = layers.Embedding(vocab_size + 1, embed_size)(context)

        # encoding the question
        query = layers.Input(shape=(query_len,), dtype='int32')
        query_encoding = layers.Embedding(vocab_size + 1, embed_size)(query)
        query_encoding = model(embed_size)(query_encoding)
        query_encoding = layers.RepeatVector(context_len)(query_encoding)

        # merging the layer
        merge = layers.add([context_encoding, query_encoding])
        merge = model(embed_size, go_backwards=True)(merge)
        prediction = layers.Dense(1, activation='sigmoid')(merge)

        model = Model([context, query], prediction)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      )

        print('Training')
        history = model.fit([data_context, data_query], answerable, batch_size=batch_size, epochs=epoch_size, verbose=1)
        with open("data/trainHistory.pickle", "wb") as history_file:
            pickle.dump(history.history, history_file)
        # save model to file
        model.save("data/LSTM_model.hdf5")