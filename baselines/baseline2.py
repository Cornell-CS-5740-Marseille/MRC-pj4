import preprocessor
import prep
import csv
import json


class Baseline2: #bigram
    def __init__(self, articles):
        self.articles = articles
        self.final_prob_answerable = 0
        self.final_prob_unanswerable = 0

    def train(self):
        prob_answerable = 0
        count_answerable = 0
        prob_unanswerable = 0
        count_unanswerable = 0
        for article in self.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    for sentence in paragraph.sentences:
                        if question.possible:
                            prob_answerable = prob_answerable + 1.0*len(set(question.words).intersection(sentence))/len(question.words)
                            count_answerable = count_answerable + 1
                        else:
                            prob_unanswerable = prob_unanswerable + 1.0 * len(
                                set(question.words).intersection(sentence)) / len(question.words)
                            count_unanswerable = count_unanswerable + 1
        self.final_prob_answerable = prob_answerable / count_answerable
        self.final_prob_unanswerable = prob_unanswerable / count_unanswerable
        print(self.final_prob_answerable)
        print(self.final_prob_unanswerable)

    def check(self, articles):
        correct = 0
        for article in articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    prob = 0
                    count = 0
                    for sentence in paragraph.sentences:
                        prob = prob + 1.0 * len(set(question.words).intersection(sentence)) / len(question.words)
                        count = count + 1
                    final_prob = prob/count
                    if final_prob < 0.192 and not question.possible or final_prob >= 0.192 and question.possible:
                        correct = correct + 1
        return correct

    def check_same_word_json(self, articles):
        out = {}
        for article in articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    prob = 0
                    count = 0
                    for sentence in paragraph.sentences:
                        prob = prob + 1.0 * len(set(question.words).intersection(sentence)) / len(question.words)
                        count = count + 1
                    final_prob = prob / count
                    if final_prob < (self.final_prob_answerable + self.final_prob_unanswerable)/2.0:
                        out[question.id] = 0
                    else:
                        out[question.id] = 1
        with open('output/output_baseline2.json', 'w') as outfile:
            json.dump(out, outfile)

    def check_same_word_csv(self, articles):
        with open('output/output_baseline2.csv', 'w') as outfile:
            speech_writer = csv.writer(outfile, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            speech_writer.writerow(['Id', 'Category'])
            for article in articles:
                for paragraph in article.paragraphs:
                    for question in paragraph.questions:
                        prob = 0
                        count = 0
                        for sentence in paragraph.sentences:
                            prob = prob + 1.0 * len(set(question.words).intersection(sentence)) / len(question.words)
                            count = count + 1
                        final_prob = prob / count
                        if final_prob < (self.final_prob_answerable + self.final_prob_unanswerable)/2.0:
                            speech_writer.writerow([question.id, 0])
                        else:
                            speech_writer.writerow([question.id, 1])


if __name__ == "__main__":
    my_prep = prep.Preprocessing(True)
    my_prep.load_file('data/training.json', False) # True: testing; False: training
    model = Baseline2(my_prep.articles)
    model.train()
    my_prep_test = prep.Preprocessing(True)
    my_prep_test.load_file('data/development.json', False)  # True: testing; False: training
    model.check_same_word_json(my_prep_test.articles)
