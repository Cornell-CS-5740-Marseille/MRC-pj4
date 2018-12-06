import preprocessor
import prep
import math
import csv
import json
import numpy
from MEMEN.word2vec import word2vec


class Baseline3:
    def __init__(self, articles):
        self.articles = articles
        self.final_cos_answerable = 0
        self.final_cos_unanswerable = 0

    def train(self):
        cos_answerable = 0
        count_answerable = 0
        cos_unanswerable = 0
        count_unanswerable = 0
        vec_model = word2vec('../GoogleNews-vectors-negative300.bin.gz')
        for article in self.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    max_q = 0
                    vec_q = numpy.zeros(300)
                    for word_q in question.words:
                        vec_q += vec_model.word_embedding(word_q)
                    vec_q = vec_q / (1.0 * len(question.words))
                    for sentence in paragraph.sentences:
                        vec_s = numpy.zeros(300)
                        for word in sentence:
                            vec_s += vec_model.word_embedding(word)
                        vec_s = vec_s / (1.0 * len(sentence))
                        cos = self.inner_product(vec_q, vec_s) / (self.get_mag(vec_q) * self.get_mag(vec_s))
                        max_q = cos if cos > max_q else max_q
                    if question.possible:
                        cos_answerable += max_q
                        count_answerable += 1
                    else:
                        cos_unanswerable += max_q
                        count_unanswerable += 1
        self.final_cos_answerable = 1.0*cos_answerable / count_answerable
        self.final_cos_unanswerable = 1.0*cos_unanswerable / count_unanswerable
        print(self.final_cos_answerable)
        print(self.final_cos_unanswerable)

        # nlp = spacy.load('en_core_web_sm')
        # doc = nlp(self.articles[0].paragraph[0].context)


    def inner_product(self, w1, w2):
        sum = 0
        for i in range(len(w1)):
            sum += w1[i]*w2[i]
        return sum

    def get_mag(self, w1):
        sum = 0
        for i in range(len(w1)):
            sum += w1[i]*w1[i]
        return math.sqrt(sum)

    def check_same_word_csv(self):
        with open('output/output_baseline3.csv', 'w') as outfile:
            speech_writer = csv.writer(outfile, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            speech_writer.writerow(['Id', 'Category'])

            vec_model = word2vec('../GoogleNews-vectors-negative300.bin.gz')
            for article in self.articles:
                for paragraph in article.paragraphs:
                    for question in paragraph.questions:
                        max_q = 0
                        vec_q = numpy.zeros(300)
                        for word_q in question.words:
                            vec_q += vec_model.word_embedding(word_q)
                        vec_q = vec_q / (1.0 * len(question.words))
                        for sentence in paragraph.sentences:
                            vec_s = numpy.zeros(300)
                            for word in sentence:
                                vec_s += vec_model.word_embedding(word)
                            vec_s = vec_s / (1.0 * len(sentence))
                            cos = self.inner_product(vec_q, vec_s) / (self.get_mag(vec_q) * self.get_mag(vec_s))
                            max_q = cos if cos > max_q else max_q
                        if cos < 0.6:
                            speech_writer.writerow([question.id, 0])
                        else:
                            speech_writer.writerow([question.id, 1])

    def check_same_word_json(self):
        out = {}
        vec_model = word2vec('../GoogleNews-vectors-negative300.bin.gz')
        for article in self.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    max_q = 0
                    vec_q = numpy.zeros(300)
                    for word_q in question.words:
                        vec_q += vec_model.word_embedding(word_q)
                    vec_q = vec_q / (1.0 * len(question.words))
                    for sentence in paragraph.sentences:
                        vec_s = numpy.zeros(300)
                        for word in sentence:
                            vec_s += vec_model.word_embedding(word)
                        vec_s = vec_s / (1.0 * len(sentence))
                        cos = self.inner_product(vec_q, vec_s) / (self.get_mag(vec_q) * self.get_mag(vec_s))
                        max_q = cos if cos > max_q else max_q
                    if cos < 0.67:
                        out[question.id] = 0
                    else:
                        out[question.id] = 1
        with open('output/output_baseline3.json', 'w') as outfile:
            json.dump(out, outfile)


# 0.6840599220488485 answerable
# 0.6749997090011657 unanswerable
if __name__ == "__main__":
    # my_prep = prep.Preprocessing(True)
    # my_prep.load_file('data/training.json', False)  # True: testing; False: training
    # model = Baseline3(my_prep.articles)
    # model.train()
    my_prep_test = prep.Preprocessing(True)
    my_prep_test.load_file('data/development.json', False)  # True: testing; False: training
    model_test = Baseline3(my_prep_test.articles)
    model_test.check_same_word_csv()
