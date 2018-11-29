import prep

class Baseline1:
    def __init__(self, articles):
        self.articles = articles
        self.threshold = 0.2

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
                        if 1.0*len(set(question.words).intersection(sentence))/len(question.words) >= self.threshold and question.possible:
                            correct = correct + 1
                            break

        return correct




if __name__ == "__main__":
    my_prep = prep.Preprocessing(True)
    my_prep.load_file('data/training.json')
    model = Baseline1(my_prep.articles)
    print(model.check_same_word()/34798/2)

