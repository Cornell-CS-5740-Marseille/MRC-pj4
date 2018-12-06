import numpy as np

from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import pickle

import re

class LSTM():
    def __init__(self):
        RNN = recurrent.LSTM
        EMBED_HIDDEN_SIZE = 100
        SENT_HIDDEN_SIZE = 100
        QUERY_HIDDEN_SIZE = 100
        BATCH_SIZE = 32
        EPOCHS = 100