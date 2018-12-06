import gensim

class word2vec():
    def __init__(self, trained_file):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(trained_file, binary=True)
        self.dim = 300
        self.dictionary = self.word2vec.wv
        self.prev_word = 'i'

    def sentence_embedding(self, sentence):
        return [self.word_embedding(word) for word in sentence]

    def word_embedding(self, word):
        if word not in self.word2vec.wv:
            return self.word2vec.wv[self.prev_word]
        else:
            self.prev_word = word
            return self.word2vec.wv[word]
