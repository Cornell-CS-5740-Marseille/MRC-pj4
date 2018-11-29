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
        self.context = ''
        self.sentences = []
        self.questions = []

    def convert_to_lower(self):
        self.context = self.context.lower()
        return self

    def parse_sentences(self):
        self.sentences = re.split('\. |; |\."|! |\? ', self.context)
        for i in range(len(self.sentences)):
            self.sentences[i] = self.sentences[i].split(' ')


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
        self.words = self.question.split(' ')

class Answer:
    def __init__(self):
        self.text = ''
        self.answer_start = 0


class Preprocessing:
    def __init__(self, lower):
        self.articles = []
        self.lower = lower

    def load_file(self, path):
        start_time = time.time()

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
                    for question in paragraph['qas']:
                        new_question = Question()
                        new_question.question = question['question'] if not self.lower else question['question'].lower()
                        new_question.id = question['id']
                        new_question.possible = question['is_impossible'] is False
                        new_question.parse_questions()
                        if new_question.possible:
                            count_possible = count_possible + 1
                        else:
                            count_impossible = count_impossible + 1
                        for answer in question['answers']:
                            new_answer = Answer()
                            new_answer.text = answer['text'] if not self.lower else answer['text'].lower()
                            new_answer.answer_start = answer['answer_start']
                            new_question.answers.append(new_answer)
                        new_paragraph.questions.append(new_question)
                    new_article.paragraphs.append(new_paragraph)
                self.articles.append(new_article)

        end_time = time.time()
        print('pre-processing time:', end_time - start_time)
        print('possible:', count_possible, 'impossible:', count_impossible)


if __name__ == "__main__":
    my_prep = Preprocessing(True)
    my_prep.load_file('data/training.json')