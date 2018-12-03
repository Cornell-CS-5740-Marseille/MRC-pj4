import preprocessor
import prep
import csv
import json
import spacy


class baseline3:
    def __init__(self):
        self.articles = articles

    def train(self):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(self.articles[0].paragraph[0].context)

if __name__ == "__main__":
    my_prep = prep.Preprocessing(True)
    my_prep.load_file('data/training.json', False)  # True: testing; False: training
    model = Baseline2(my_prep.articles)
    model.train()
    # my_prep_test = prep.Preprocessing(True)
    # my_prep_test.load_file('data/testing.json', True)  # True: testing; False: training
    # model.check_same_word_csv(my_prep_test.articles)