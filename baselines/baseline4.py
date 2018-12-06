import prep
import csv
import json

class Baseline4:
    def __init__(self, articles, NER_text, NER_tag):
        self.articles = articles
        self.NER_text = NER_text
        self.NER_tag = NER_tag
        self.threshold_NER = 0.3
        self.threshold_SIM = 0.4

    def check_same_word_csv(self):
        count = 0
        tmp = 0
        with open('output/output_baseline4.csv', 'w') as outfile:
            speech_writer = csv.writer(outfile, delimiter=',', quotechar='"',
                                       quoting=csv.QUOTE_MINIMAL)
            speech_writer.writerow(['Id', 'Category'])
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
                            if ('country' or 'city' or 'state' in question.words) and ('GPE' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and NER_text[
                                NER_tag[tmp].index('GPE')] not in question.words:
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('who' in question.words) and ('PERSON' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER and NER_text[
                                         NER_tag[tmp].index('PERSON')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('where' in question.words) and ('LOC' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('LOC')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('where' in question.words) and ('GPE' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('GPE')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('where' in question.words) and ('FAC' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('FAC')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('when' or 'date' or 'time' in question.words) and ('DATE' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('DATE')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('when' or 'date' or 'time' in question.words) and ('TIME' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('TIME')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('how' and 'many' in question.words) and ('CARDINAL' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('CARDINAL')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('how' and 'many' in question.words) and ('QUANTITY' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and \
                                    (NER_text[NER_tag[tmp].index('QUANTITY')] not in question.words):
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('how' and 'much' in question.words) and ('MONEY' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and NER_text[
                                NER_tag[tmp].index('MONEY')] not in question.words:
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('per' in question.words) and ('PERCENT' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and NER_text[
                                NER_tag[tmp].index('PERCENT')] not in question.words:
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif ('work' or 'job' in question.words) and ('ORG' in self.NER_tag[tmp]) and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and NER_text[
                                NER_tag[tmp].index('ORG')] not in question.words:
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif 'national' in question.words and 'NORP' in self.NER_tag[tmp] and \
                                    (1.0 * len(set(question.words).intersection(sentence)) / len(
                                        question.words) >= self.threshold_NER) and NER_text[
                                NER_tag[tmp].index('NORP')] not in question.words:
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            elif 1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_SIM:
                                tmp = tmp + 1
                                speech_writer.writerow([question.id, 1])
                                found = True
                                break
                            else:
                                tmp = tmp + 1
                        if not found:
                            speech_writer.writerow([question.id, 0])
                    count = count + len(paragraph.sentences)


    def check_same_word_json(self):
        out = {}
        # count = 0
        count = 0
        tmp = 0
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
                        if ('country' or 'city' or 'state' in question.words) and ('GPE' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and NER_text[NER_tag[tmp].index('GPE')] not in question.words:
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('who' in question.words) and ('PERSON' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER and NER_text[NER_tag[tmp].index('PERSON')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('where' in question.words) and ('LOC' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('LOC')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('where' in question.words) and ('GPE' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('GPE')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('where' in question.words) and ('FAC' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('FAC')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('when' or 'date' or 'time' in question.words) and ('DATE' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('DATE')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('when' or 'date' or 'time' in question.words) and ('TIME' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('TIME')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('how' and 'many' in question.words) and ('CARDINAL' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('CARDINAL')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('how' and 'many' in question.words) and ('QUANTITY' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and \
                                (NER_text[NER_tag[tmp].index('QUANTITY')] not in question.words):
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('how' and 'much' in question.words) and ('MONEY' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and NER_text[NER_tag[tmp].index('MONEY')] not in question.words:
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('per' in question.words) and ('PERCENT' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and NER_text[
                            NER_tag[tmp].index('PERCENT')] not in question.words:
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif ('work' or 'job' in question.words) and ('ORG' in self.NER_tag[tmp]) and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and NER_text[
                            NER_tag[tmp].index('ORG')] not in question.words:
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif 'national' in question.words and 'NORP' in self.NER_tag[tmp] and \
                                (1.0 * len(set(question.words).intersection(sentence)) / len(
                                    question.words) >= self.threshold_NER) and NER_text[
                            NER_tag[tmp].index('NORP')] not in question.words:
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        elif 1.0 * len(set(question.words).intersection(sentence)) / len(question.words) >= self.threshold_SIM:
                            tmp = tmp + 1
                            out[question.id] = 1
                            found = True
                            break
                        else:
                            tmp = tmp + 1
                    if not found:
                        out[question.id] = 0
                count = count + len(paragraph.sentences)

        with open('output/output_baseline4.json', 'w') as outfile:
            json.dump(out, outfile)
        print(len(out))


if __name__ == "__main__":
    my_prep = prep.Preprocessing(True)
    my_prep.load_file('data/development.json', False)  #validation
    fhandle = open('data/NER_tag_validation.txt', 'r')  #validation
    
    #my_prep.load_file('data/test.json', True)   #test
    #fhandle = open('data/NER_tag_test.txt', 'r')  #test
    NER_tag = []
    NER_text = []
    # i = 0
    for line in fhandle:
        tag = line.strip('\n').split(',')
        if tag[0] == '0':
            NER_text.append(tag)
        elif tag[0] == '1':
            NER_tag.append(tag)
        # print(NER[i])
        # i = i + 1
    print(len(NER_text))
    print(len(NER_tag))
    model = Baseline4(my_prep.articles,NER_text,NER_tag)


    model.check_same_word_csv()


    model.check_same_word_json()



