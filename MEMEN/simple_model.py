import pickle

from keras import layers, Sequential, Input
from keras.layers import LSTM
from keras.models import Model

class MemNN():
    def __init__(self, context_len, vocab_size, embed_size, query_len, batch_size, epoch_size,
                 data_context, data_query, data_answer, answerable, dropout=0.3):
        context_sequence = Input((context_len,))
        query_sequence = Input((query_len,))

        # encoding the passage
        input_encoder_m = Sequential()
        input_encoder_m.add(layers.Embedding(input_dim=vocab_size,
                                      output_dim=embed_size))
        input_encoder_m.add(layers.Dropout(dropout))

        # embed the input into a sequence of vectors of size query_len
        input_encoder_c = Sequential()
        input_encoder_c.add(layers.Embedding(input_dim=vocab_size,
                                      output_dim=query_len))

        # embed the question into a sequence of vectors
        question_encoder = Sequential()
        question_encoder.add(layers.Embedding(input_dim=vocab_size,
                                       output_dim=embed_size,
                                       input_length=query_len))
        question_encoder.add(layers.Dropout(dropout))
        # output: (samples, query_maxlen, embedding_dim)

        # encode input sequence and questions (which are indices)
        # to sequences of dense vectors
        input_encoded_m = input_encoder_m(context_sequence)
        input_encoded_c = input_encoder_c(context_sequence)
        question_encoded = question_encoder(query_sequence)

        # compute a 'match' between the first input vector sequence
        # and the question vector sequence
        # shape: `(samples, story_maxlen, query_maxlen)`
        match = layers.dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = layers.Activation('softmax')(match)

        # add the match matrix with the second input vector sequence
        response = layers.add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
        response = layers.Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

        # concatenate the match matrix with the question vector sequence
        answer = layers.concatenate([response, question_encoded])

        # the original paper uses a matrix multiplication for this reduction step.
        # we choose to use a RNN instead.
        answer = LSTM(32)(answer)  # (samples, 32)

        # one regularization layer -- more would probably be needed.
        answer = layers.Dropout(dropout)(answer)
        answer = layers.Dense(1)(answer)  # (samples, vocab_size)
        # we output a probability distribution over the vocabulary
        answer = layers.Activation('sigmoid')(answer)

        # build the final model
        model = Model([context_sequence, query_sequence], answer)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

        print('Training')
        # train
        history = model.fit([data_context, data_query], answerable,
                  batch_size=batch_size,
                  epochs=epoch_size, verbose=1)

        with open("data/trainHistory.pickle", "wb") as history_file:
            pickle.dump(history.history, history_file)
        # save model to file
        model.save("data/MEMNN_model.hdf5")