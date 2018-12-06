from NER import linguistic_feature
from prep_NER import Preprocessing
import string
import json


text = []
NER = []
POS = []
ner_processor = linguistic_feature()
my_prep = Preprocessing(True)
my_prep.load_file('data/testing.json', True) # True: testing; False: training
fhandle1 = open("data/POS_tag_testing_final.txt",'w')
fhandle2 = open("data/NER_tag_testing_final.txt",'w')
for article in my_prep.articles:
    for paragraph in article.paragraphs:
        for i in range(len(paragraph.sentences)):
            POS, NER_text, NER_tag = ner_processor.load_text(paragraph.sentences[i])
            fhandle1.write(",".join(POS))
            fhandle1.write('\n')
            text = ",".join(NER_text)
            fhandle2.write(str(0)+',')
            fhandle2.write(text)
            fhandle2.write('\n')
            tag = ",".join(NER_tag)
            fhandle2.write(str(1) + ',')
            fhandle2.write(tag)
            fhandle2.write('\n')




