import prep
import csv
import json

class Baseline4:
    def __init__(self, articles, NER):
        self.articles = articles
        self.NER = NER
        self.threshold = 0.2

    def check_same_word(self):
        correct = 0
        count = 1
        for article in self.articles:
            for paragraph in article.paragraphs:
                for question in paragraph.questions:
                    found = False
                    for sentence in paragraph.sentences:
                        # print(set(question.words).intersection(sentence))
                        # print(sentence)
                        # print(question.words)
                        # print(len(set(question.words).intersection(sentence)), len(question.words))
                        # print(question.words)
                        if count > len(NER)-1:
                            return correct,count
                        if NER[count][0] != '1':
                            count = count + 1
                            continue
                        print(NER[count])
                        if ('country' or 'city' or 'state' in question.words) and ('GPE' in self.NER[count]) and \
                                (1.0*len(set(question.words).intersection(sentence))/len(question.words) >= self.threshold and question.possible):
                            correct = correct + 1
                            count = count + 1
                            found = True
                            break
                        elif ('who' in question.words) and ('PERSON' in self.NER[count]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(question.words) >= self.threshold and question.possible):
                            correct = correct + 1
                            count = count + 1
                            found = True
                            break
                        elif ('where' in question.words) and ('LOC' or 'GPE' or 'FAC' in self.NER[count]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(question.words) >= self.threshold and question.possible):
                            correct = correct + 1
                            count = count + 1
                            found = True
                            break
                        elif ('data' or 'time' in question.words) and ('DATE' or 'TIME' in self.NER[count]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(question.words) >= self.threshold and question.possible):
                            correct = correct + 1
                            count = count + 1
                            found = True
                            break
                        elif 1.0*len(set(question.words).intersection(sentence))/len(question.words) >= self.threshold and question.possible:
                            correct = correct + 1
                            count = count + 2
                            found = True
                            break
                        else:
                            count = count + 1
                    if not found and not question.possible:
                        correct = correct + 1


        return correct, count

    def check_same_word_csv(self):
        count = 1
        tmp = 1
        with open('output/output_baseline4.csv', 'w') as outfile:
            speech_writer = csv.writer(outfile, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            speech_writer.writerow(['Id', 'Predicted'])
            for article in self.articles:
                for paragraph in article.paragraphs:
                    for question in paragraph.questions:
                        found = False
                        tmp = count
                        for sentence in paragraph.sentences:
                            # print(set(question.words).intersection(sentence))
                            # print(sentence)
                            # print(question.words)
                            # print(len(set(question.words).intersection(sentence)), len(question.words))
                            # print(question.words)
                            # print(NER[count])
                            if ('country' or 'city' or 'state' in question.words) and ('GPE' in self.NER[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold):
                                tmp = tmp + 2
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('who' in question.words) and ('PERSON' in self.NER[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold):
                                tmp = tmp + 2
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('where' in question.words) and ('LOC' or 'GPE' or 'FAC' in self.NER[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold):
                                tmp = tmp + 2
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('data' or 'time' in question.words) and ('DATE' or 'TIME' in self.NER[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold):
                                tmp = tmp + 2
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif 1.0 * len(set(question.words).intersection(sentence)) / len(question.words) >= self.threshold:
                                tmp = tmp + 2
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            else:
                                tmp = tmp + 2
                        if not found:
                            speech_writer.writerow([question.id, 0])
                    count = count + len(paragraph.sentences)*2

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
    my_prep.load_file('data/testing.json', True)

    fhandle = open('data/NER_tag_testing_final.txt', 'r')
    NER = []
    # i = 0
    for line in fhandle:
        tag = line.strip('\n').split(',')
        NER.append(tag)
        # print(NER[i])
        # i = i + 1

    model = Baseline4(my_prep.articles,NER)
    # correct,count = model.check_same_word()
    # print(correct,count)
    # print(correct/((count-1)/2))
    # print(model.check_same_word()/69596)
    model.check_same_word_csv()

