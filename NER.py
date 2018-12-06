import spacy

class linguistic_feature():

    def load_text(self, text):
        POS = []
        NER_tag = []
        NER_text = []
        nlp = spacy.load('en_core_web_sm')
        for line in text:
            doc = nlp(line)
            for token in doc:
                POS.append(token.tag_)
            for ent in doc.ents:
                NER_tag.append(ent.label_)
                NER_text.append(ent.text)
        return POS, NER_text, NER_tag


if __name__ == "__main__":
    processor = linguistic_feature()
    processor.load_text(u'Beyonce Giselle Knowles-Carter')