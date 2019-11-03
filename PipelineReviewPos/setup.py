import pickle
import json
import collections
import spacy
from metal.contrib.info_extraction.mentions import RelationMention
import numpy as np
import progressbar
from collections import Counter



DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/dev_examples.pkl'
DATA_FILE3 = 'data/test_examples.pkl'
DATA_FILE4 = 'data/train_labels.pkl'
DATA_FILE5 = 'data/dev_labels.pkl'
DATA_FILE6 = 'data/test_labels.pkl'
DATA_FILE7 = 'data/data.pkl'
DATA_FILE8 = 'data/labels.pkl'
DATA_FILE9 = 'data/nouns_list.pkl'


def setup():
    train_list = []
    dev_list = []
    test_list = []

    print("Reading from csv and splitting")
    for i, line in enumerate(open('../data/reviews200k.json', 'r')):
        if i < 180000 and len(line) < 500:
            train_list.append(json.loads(line))
        if 179999 < i < 182500 and len(line) < 500:
            dev_list.append(json.loads(line))
        if 184999 < i < 187500 and len(line) < 500:
            test_list.append(json.loads(line))

    print(len(train_list))
    train_reviews = []
    for i in range(0, 6):
        j = 0
        for line in train_list:
            if i == int(line['stars']):
                train_reviews.append(line)
                j = j + 1
            if j > 499:
                break

    train_examples = [review['text'].lower() for review in train_reviews]
    train_labels = [int(review['stars']) for review in train_reviews]

    print(collections.Counter(train_labels))

    with open(DATA_FILE1, 'wb') as f:
        pickle.dump(train_examples, f)
    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(train_labels, f)

    dev_examples = [review['text'].lower() for review in dev_list]
    dev_labels = [int(review['stars']) for review in dev_list]

    with open(DATA_FILE2, 'wb') as f:
        pickle.dump(dev_examples, f)

    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(dev_labels, f)

    test_examples = [review['text'].lower() for review in test_list]
    test_labels = [int(review['stars']) for review in test_list]

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(test_examples, f)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(test_labels, f)

    print("Done")

    print("Extracting nouns list")

    spacy_nlp = spacy.load('en_core_web_sm')

    nouns_list = []

    for example in progressbar.progressbar(train_examples):
        doc = spacy_nlp(example)

        nouns = [token.text for token in doc if not token.is_stop and token.is_alpha and token.tag_ in 'NN'
                 and not token.ent_type_ in ('PERSON', 'GPE', 'NORP', 'DATE', 'CARDINAL')]
        nouns_list = nouns_list + nouns

    nouns_ordered_list = [item for items, c in Counter(nouns_list).most_common()
                          for item in [items] * c]
    nouns_list = list(dict.fromkeys(nouns_ordered_list))

    nouns_list = nouns_list[0:2000]

    with open(DATA_FILE9, 'wb') as f:
        pickle.dump(nouns_list, f)

    print("Creating objects")
    train_results_examples = []
    train_results_labels = []
    dev_results_examples = []
    dev_results_labels = []
    test_results_examples = []
    test_results_labels = []
    index = 1


    for example, label in progressbar.progressbar(zip(train_examples, train_labels)):
        doc = spacy_nlp(example)

        words, char_offsets, pos_tags, ner_tags, entity_types, entitiesId = ([] for i in range(6))
        for sent in doc.sents:
            for i, token in enumerate(sent):
                words.append(str(token))
                pos_tags.append(token.tag_)
                ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
                char_offsets.append(token.idx)
                entity_types.append('O')
                if token.text in nouns_list and len(entitiesId) < 3:
                    start = int(token.idx)
                    end = int(token.idx + len(token.text))
                    entitiesId.append((start, end))

        if len(entitiesId) == 3:
            result = RelationMention(index, example, entitiesId
                                     , words, char_offsets, pos_tags=pos_tags, ner_tags=ner_tags,
                                     entity_types=entity_types)
            train_results_examples.append(result)
            train_results_labels.append(label)
            index = index + 1

    for example, label in progressbar.progressbar(zip(dev_examples, dev_labels)):
        doc = spacy_nlp(example)

        words, char_offsets, pos_tags, ner_tags, entity_types, entitiesId = ([] for i in range(6))
        for sent in doc.sents:
            for i, token in enumerate(sent):
                words.append(str(token))
                pos_tags.append(token.tag_)
                ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
                char_offsets.append(token.idx)
                entity_types.append('O')
                if token.text in nouns_list and len(entitiesId) < 3:
                    start = int(token.idx)
                    end = int(token.idx + len(token.text))
                    entitiesId.append((start, end))

        if len(entitiesId) == 3:
            result = RelationMention(index, example, entitiesId
                                     , words, char_offsets, pos_tags=pos_tags, ner_tags=ner_tags,
                                     entity_types=entity_types)
            dev_results_examples.append(result)
            dev_results_labels.append(label)
            index = index + 1

    for example, label in progressbar.progressbar(zip(test_examples, test_labels)):
        doc = spacy_nlp(example)

        words, char_offsets, pos_tags, ner_tags, entity_types, entitiesId = ([] for i in range(6))
        for sent in doc.sents:
            for i, token in enumerate(sent):
                words.append(str(token))
                pos_tags.append(token.tag_)
                ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
                char_offsets.append(token.idx)
                entity_types.append('O')
                if token.text in nouns_list and len(entitiesId) < 3:
                    start = int(token.idx)
                    end = int(token.idx + len(token.text))
                    entitiesId.append((start, end))

        if len(entitiesId) == 3:
            result = RelationMention(index, example, entitiesId
                                     , words, char_offsets, pos_tags=pos_tags, ner_tags=ner_tags,
                                     entity_types=entity_types)
            test_results_examples.append(result)
            test_results_labels.append(label)
            index = index + 1

    Cs = [train_results_examples, dev_results_examples, dev_results_examples]

    Ys = [np.array(train_results_labels), np.array(dev_results_labels), np.array(test_results_labels)]

    with open(DATA_FILE7, 'wb') as f:
        pickle.dump(Cs, f)

    with open(DATA_FILE8, 'wb') as f:
        pickle.dump(Ys, f)

    print("Done")
