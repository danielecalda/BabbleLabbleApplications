import pickle
import json
import collections
import spacy
from metal.contrib.info_extraction.mentions import RelationMention
import numpy as np
import progressbar


DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/dev_examples.pkl'
DATA_FILE3 = 'data/test_examples.pkl'
DATA_FILE4 = 'data/train_labels.pkl'
DATA_FILE5 = 'data/dev_labels.pkl'
DATA_FILE6 = 'data/test_labels.pkl'
DATA_FILE7 = 'data/data.pkl'
DATA_FILE8 = 'data/labels.pkl'


def setup():
    train_list = []
    dev_list = []
    test_list = []

    print("Reading from csv and splitting")
    for i, line in enumerate(open('../data/reviews200k.json', 'r')):
        if i < 180000:
            train_list.append(json.loads(line))
        if 179999 < i < 181000:
            dev_list.append(json.loads(line))
        if 180999 < i < 182000:
            test_list.append(json.loads(line))

    print(len(train_list))
    train_reviews = []
    for i in range(0, 6):
        j = 0
        for line in train_list:
            if i == int(line['stars']):
                train_reviews.append(line)
                j = j + 1
            if j > 999:
                break

    train_examples = [review['text'] for review in train_reviews]
    train_labels = [review['stars'] for review in train_reviews]

    print(collections.Counter(train_labels))

    with open(DATA_FILE1, 'wb') as f:
        pickle.dump(train_examples, f)
    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(train_labels, f)

    dev_examples = [review['text'] for review in dev_list]
    dev_labels = [review['stars'] for review in dev_list]

    with open(DATA_FILE2, 'wb') as f:
        pickle.dump(dev_examples, f)

    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(dev_labels, f)

    test_examples = [review['text'] for review in test_list]
    test_labels = [review['stars'] for review in test_list]

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(test_examples, f)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(test_labels, f)

    print("Done")

    print("Creating objects")
    train_results = []

    spacy_nlp = spacy.load('en_core_web_sm')

    for example in progressbar.progressbar(train_examples):
        doc = spacy_nlp(example)

        words, char_offsets, pos_tags, ner_tags, entity_types = ([] for i in range(5))
        for sent in doc.sents:
            for i, token in enumerate(sent):
                words.append(str(token))
                pos_tags.append(token.tag_)
                ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
                char_offsets.append(token.idx)
                entity_types.append('O')

        result = RelationMention(1, example, [(0, 2), (4, 5)], words, char_offsets, pos_tags=pos_tags,
                                 ner_tags=ner_tags, entity_types=entity_types)

        train_results.append(result)

    dev_results = []

    for example in progressbar.progressbar(dev_examples):
        doc = spacy_nlp(example)

        words, char_offsets, pos_tags, ner_tags, entity_types = ([] for i in range(5))
        for sent in doc.sents:
            for i, token in enumerate(sent):
                words.append(str(token))
                pos_tags.append(token.tag_)
                ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
                char_offsets.append(token.idx)
                entity_types.append('O')

        result = RelationMention(1, example, [(0, 2), (4, 5)], words, char_offsets, pos_tags=pos_tags,
                                 ner_tags=ner_tags, entity_types=entity_types)

        dev_results.append(result)

    test_results = []

    for example in progressbar.progressbar(test_examples):
        doc = spacy_nlp(example)

        words, char_offsets, pos_tags, ner_tags, entity_types = ([] for i in range(5))
        for sent in doc.sents:
            for i, token in enumerate(sent):
                words.append(str(token))
                pos_tags.append(token.tag_)
                ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
                char_offsets.append(token.idx)
                entity_types.append('O')

        result = RelationMention(1, example, [(0, 2), (4, 5)], words, char_offsets, pos_tags=pos_tags,
                                 ner_tags=ner_tags, entity_types=entity_types)

        test_results.append(result)

    Cs = [train_results, dev_results, test_results]

    Ys = [np.array(train_labels), np.array(dev_labels), np.array(test_labels)]

    with open(DATA_FILE7, 'wb') as f:
        pickle.dump(Cs, f)

    with open(DATA_FILE8, 'wb') as f:
        pickle.dump(Ys, f)

    print("Done")
