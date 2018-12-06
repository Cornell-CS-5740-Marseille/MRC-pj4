import prep
import csv
import json


class Baseline1:
    def __init__(self, articles):
        self.articles = articles
        self.threshold = 0.3

    def check_same_word(self):
        correct = 0
        for article in self.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    for sentence in paragraph.sentences:
                        # print(set(question.words).intersection(sentence))
                        # print(sentence)
                        # print(question.words)
                        # print(len(set(question.words).intersection(sentence)), len(question.words))
                        if 1.0*len(set(question.words).intersection(sentence))/len(question.words) >= self.threshold and question.possible or \
                            1.0 * len(set(question.words).intersection(sentence)) / len(question.words) < self.threshold and not question.possible:
                            correct = correct + 1
                            break

        return correct

    def check_same_word_csv(self):
        with open('output/output_baseline1.csv', 'w') as outfile:
            speech_writer = csv.writer(outfile, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            speech_writer.writerow(['Id', 'Category'])
            for article in self.articles:
                for paragraph in article.paragraphs:
                    for question in paragraph.questions:
                        found = False
                        for sentence in paragraph.sentences:
                            if 1.0*len(set(question.words).intersection(sentence))/len(question.words) >= self.threshold:
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                        if not found:
                            speech_writer.writerow([question.id, 0])

    def check_same_word_json(self):
        out = {}
        count = 0
        for article in self.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    found = False
                    for sentence in paragraph.sentences:

                        # print(set(question.words).intersection(sentence))
                        # print(sentence)
                        # print(question.words)
                        # print(len(set(question.words).intersection(sentence)), len(question.words))
                        if 1.0 * len(set(question.words).intersection(sentence)) / len(
                                question.words) >= self.threshold:
                            count = count + 1
                            out[question.id] = 1
                            found = True
                            break
                    if not found:
                        out[question.id] = 0
        with open('output/output_baseline1.json', 'w') as outfile:
            json.dump(out, outfile)
        print(len(out))


if __name__ == "__main__":
    my_prep = prep.Preprocessing(True)
    my_prep.load_file('data/development.json', False)
    model = Baseline1(my_prep.articles)
    model.check_same_word_json()
    # print(model.check_same_word()/69596)

