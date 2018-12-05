import json
import time
import re


class Article:
    def __init__(self):
        self.title = ''
        self.paragraphs = []

    def convert_to_lower(self):
        self.title = self.title.lower()
        return self


class Paragraph:
    def __init__(self):
        self.sentences_word = []
        self.context = ''
        self.sentences = []
        self.questions = []
        self.words = []

    def convert_to_lower(self):
        self.context = self.context.lower()
        return self

    def parse_sentences(self):
        self.words = re.split(' ', self.context)
        self.sentences = re.split('\. |; |\."|! |\? ', self.context)
        for i in range(len(self.sentences)):
            self.sentences[i] = self.sentences[i].split(' ')
            self.sentences_word += self.sentences[i]


class Question:
    def __init__(self):
        self.question = ''
        self.id = ''
        self.answers = []
        self.words = []
        self.possible = False

    def convert_to_lower(self):
        self.question = self.question.lower()
        return self

    def parse_questions(self):
        self.words = re.split(' ', self.question)

class Answer:
    def __init__(self):
        self.text = ''
        self.answer_start = 0
        self.words = []


class Preprocessing:
    def __init__(self, lower):
        self.articles = []
        self.dictionary = {}
        self.context_words = []
        self.lower = lower
        self.dict_index = 0

    def load_file(self, path, is_test):
        start_time = time.time()

        count_total = 0
        count_possible = 0
        count_impossible = 0
        with open(path, 'r') as f:
            data = json.load(f)['data']
            for article in data:
                new_article = Article()
                new_article.title = article['title'] if not self.lower else article['title'].lower()
                for paragraph in article['paragraphs']:
                    new_paragraph = Paragraph()
                    new_paragraph.context = paragraph['context'] if not self.lower else paragraph['context'].lower()
                    new_paragraph.parse_sentences()
                    self.update_dict(new_paragraph.words)
                    self.context_words.append(new_paragraph.words)
                    for question in paragraph['qas']:
                        new_question = Question()
                        new_question.question = question['question'] if not self.lower else question['question'].lower()
                        new_question.id = question['id']
                        new_question.parse_questions()
                        self.update_dict(new_question.words)

                        count_total = count_total + 1
                        if not is_test:
                            new_question.possible = question['is_impossible'] is False
                            if new_question.possible:
                                count_possible = count_possible + 1
                            else:
                                count_impossible = count_impossible + 1
                            for answer in question['answers']:
                                new_answer = Answer()
                                new_answer.text = answer['text'] if not self.lower else answer['text'].lower()
                                new_answer.answer_start = answer['answer_start']
                                new_answer.words = re.split(' ', new_answer.text)
                                self.update_dict(new_answer.words)
                                new_question.answers.append(new_answer)
                        new_paragraph.questions.append(new_question)
                    new_article.paragraphs.append(new_paragraph)
                self.articles.append(new_article)

        end_time = time.time()
        print('pre-processing time:', end_time - start_time)
        print('total:', count_total, 'possible:', count_possible, 'impossible:', count_impossible)

    def update_dict(self, sentence):
        for word in sentence:
            if not (word in self.dictionary):
                self.dictionary[word] = self.dict_index
                self.dict_index = self.dict_index + 1

    def word_ids(self, sentence):
        return list(map(lambda x: self.dictionary[x], sentence))


if __name__ == "__main__":
    my_prep = Preprocessing(True)
    my_prep.load_file('data/training.json', False)