import math
import random
import json
from functools import reduce

from src.preprocessor import preprocessor


class Ngrams:
    # Initialization method
    def __init__(self, opts):
        self.n = opts["n"]  # length of n-gram (trigram: n=3)
        self.start_symbol = "<s>"  # start of document
        self.end_symbol = "</s>"  # end of document
        self.unknown_symbol = "<u>"  # unknown n-gram
        self.placeholder = " "  # buffer used in case of short prefix (?)
        self.count_table = {}   # data structure for the n-gram count; dict<string, dict<string, double>>
        self.threshold = opts["threshold"]  # max number of words in teh sentence generation

        # for smoothing
        self.all_count_symbol = "<a>"  # counter for each key
        self.other_symbol = "<o>"  # unknown n-gram
        self.all_unique_pair_counter = 0  # counter for all n-gram pairs
        self.discount = 0.1  # discount value to be used in kneser-ney
        self.all_tokens = set({self.start_symbol, self.end_symbol})
        self.smoothed_count_table = {}  # data structure for the smoothed n-gram count;
                                        # dict<string, dict<string, double>>
        self.reverse_dict = {}

    def to_key(self, prefix):               # convert a string to a dictionary key
        return "-".join(prefix)

    def generate_count_table(self, logs):
        n = self.n
        words = [logs[3]]
        # add each occurrence of n-gram into the dictionary. The key is the
        # (n-1)-gram and the value is another dictionary of each words' frequency
        # following that (n-1)-gram
        for log in words:
            for i in range(0, len(log) - n + 1):
                if n > 1:
                    key = self.to_key(log[i: i + n - 1])
                else:
                    key = self.placeholder
                target = log[i + n - 1]
                if key in self.count_table:
                    if target in self.count_table[key]:
                        self.count_table[key][target] += 1
                    else:
                        self.count_table[key][target] = 1
                else:
                    self.count_table[key] = {target: 1}
        return self.count_table

    def convert_count_to_prob(self):
        for key in self.count_table:
            count_dict = self.count_table[key]
            number = reduce((lambda x, y: x + y), count_dict.values())
            for target in count_dict:
                count_dict[target] /= (number + 0.0)
        return self.count_table

    # create the count table with raw frequencies at first
    def dist_table_unsmoothed(self, logs):
        self.generate_count_table(logs)
        self.convert_count_to_prob()
        return self.count_table

    def dist_table_add_one_smooth(self, logs, k):
        self.generate_count_table(logs)
        for key in self.count_table:
            for target in logs[2]:
                if target in self.count_table[key]:
                    self.count_table[key][target] += k
                else:
                    self.count_table[key][target] = k
        ###print self.count_table
        self.convert_count_to_prob()
        return self.count_table

    def dist_table_smoothed_kneser_ney(self, logs):
        words = [logs[3]]
        n = self.n
        for log in words:
            for i in range(0, len(log) - n + 1):
                if n > 1:
                    key = self.to_key(log[i: i + n - 1])
                else:
                    key = self.placeholder
                target = log[i + n - 1]
                self.all_tokens.add(target)

                # count table
                if key in self.count_table:
                    if target in self.count_table[key]:
                        self.count_table[key][target] += 1
                    else:
                        self.count_table[key][target] = 1
                    self.count_table[key][self.all_count_symbol] += 1
                else:
                    self.count_table[key] = {target: 1}
                    self.count_table[key][self.all_count_symbol] = 1

                # reverse count table
                if target in self.reverse_dict:
                    if key in self.reverse_dict[target]:
                        self.reverse_dict[target][key] += 1
                    else:
                        self.reverse_dict[target][key] = 1
                    self.reverse_dict[target][self.all_count_symbol] += 1
                else:
                    self.reverse_dict[target] = {key: 1}
                    self.reverse_dict[target][self.all_count_symbol] = 1

        for key in self.count_table:
            self.all_unique_pair_counter += len(self.count_table[key]) - 1  # minus the <a> in the dict

        for key in self.count_table:
            self.smoothed_count_table[key] = {}
            count_dict = self.count_table[key]
            for target in logs[2]:
                if target not in logs[1]:
                    if target != self.all_count_symbol:
                        count = 0
                        if target in count_dict:
                            count = count_dict[target]
                        percentage_after_discount = max(count - self.discount, 0) / \
                                                    (count_dict[self.all_count_symbol] + 0.0)
                        normalized = self.discount * len(count_dict) / (count_dict[self.all_count_symbol] + 0.0)
                        reverse_count = 0
                        if target in self.reverse_dict:
                            reverse_count = len(self.reverse_dict[target]) - 1
                        prev = reverse_count / (self.all_unique_pair_counter + 0.0)
                        smoothed_prob = percentage_after_discount + normalized * prev
                        self.smoothed_count_table[key][target] = smoothed_prob
        self.count_table = self.smoothed_count_table
        return self.count_table

    def lookup_dist(self, sequence, target):
        n = self.n
        key = self.to_key(sequence) if n > 1 else self.placeholder
        if key in self.count_table:
            if target in self.count_table[key]:
                return self.count_table[key][target]
            else:
                ###print "no target:", target
                return self.count_table[key][self.unknown_symbol]
        else:
            ###print "no key:", key
            if target in self.count_table[self.unknown_symbol]:
                return self.count_table[self.unknown_symbol][target]
            else:
                ###print "no target:", target
                return self.count_table[self.unknown_symbol][self.unknown_symbol]

    def unsmoothed_nGram(self, sentence):
        sentence = sentence.split(" ")
        n = self.n
        probability = 1
        for i in range(0, len(sentence) - n + 1):
            sequence = sentence[i: i + n - 1] if n > 1 else []
            target = sentence[i + n - 1]
            prob = self.lookup_dist(sequence, target)
            probability *= prob

        return probability

    def sentence(self, prefix):
        # assume that length of prefix is larger than n
        sentence = prefix
        prefix = prefix.split(' ') if len(prefix) > 0 else []
        if len(prefix) >= self.n - 1:
            start = prefix[len(prefix) - self.n:]
        else:
            start = ["<s>"]
        for x in range(0, self.threshold):
            key = self.to_key(start) if self.n > 1 else self.placeholder
            if key in self.count_table:
                prob = random.uniform(0, 1)
                items = self.count_table[key].items()
                lower = 0
                for target, probability in items:
                    if prob >= lower and prob < probability + lower:
                        sentence = sentence + " " + target
                        start.append(target)
                        start = start[1:]
                        break
                    else:
                        lower += probability
            else:
                ###print "no key:", key
                return sentence
        return sentence

    def perplexity(self, test_list):
        exponent = 0
        n = self.n
        for x in range(0, len(test_list) - n + 1):
            keys = (test_list[x: x + n - 1]) if n > 1 else []
            target = test_list[x + n - 1]
            if (self.lookup_dist(keys, target) > 0):
                exponent = exponent - math.log(self.lookup_dist(keys, target))

        exponent /= (len(test_list) - n + 1.0)
        PP = math.exp(exponent)
        return PP

    def save_model(self, file):
        fileName = file + "-" + str(self.n) + ".json"
        with open(fileName, 'w') as fp:
            json.dump(self.count_table, fp)


# test cases
# corpus = ["<s> I am Sam </s>".split(" "), "<s> Sam I am </s>".split(" "), "<s> I do not like green eggs and ham </s>".split(" ")]
# ngram = Ngrams({"n": 2, "threshold": 100})
# ###print ngram.dist_table(corpus)
# ###print ngram.sentence('I')

#data = preprocessor("../Assignment1_resources/train/trump.txt",0).data
#ngram = Ngrams({"n": 1, "threshold": 100})
#ngram.dist_table_unsmoothed(data[0])
# ngram.dist_table_smoothed_kneser_ney(data)
#ngram.dist_table_add_one_smooth(data, 0.1)
# ###print data[1]
# ###print data[2]
# ###print data[3]
# ###print ngram.count_table
#test1 = preprocessor("../Assignment1_resources/development/trump.txt", 0).data[0]
#test2 = preprocessor("../Assignment1_resources/development/obama.txt", 0).data[0]
# test = preprocessor("../Assignment1_resources/development/small_test.txt").data[0]

#print ngram.perplexity(test2)
#print ngram.perplexity(test1)

# ngram.save_model("model/obama")
# ###print ngram.sentence('i')

