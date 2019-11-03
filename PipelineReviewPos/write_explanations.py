import pickle
import progressbar
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from babble import Explanation
from babble.utils import ExplanationIO

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE3 = 'data/pos_adjectives_list.pkl'
DATA_FILE4 = 'data/neu_adjectives_list.pkl'
DATA_FILE5 = 'data/neg_adjectives_list.pkl'
DATA_FILE6 = "data/my_explanations.tsv"

pos_adjectives_list = []
neu_adjectives_list = []
neg_adjectives_list = []


def write_explanations():
    print("Writing explanations")

    with open(DATA_FILE1, 'rb') as f:
        reviews = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        labels = pickle.load(f)

    entity_names = ['X', 'Y', 'Z']

    explanations = []
    index = 1

    for candidate, label in progressbar.progressbar(zip(reviews[0], labels[0])):
        #    print(candidate.text)
        for entity, name in zip(candidate.entities, entity_names):
            #        print(entity.entity)
            adjective = check_adjectives_after_verb(candidate.text, entity.entity)
            sentiment_value = ''
            if adjective is not None:
                sentiment_value = check_sentiment_adjective(adjective)
                condition = 'A ' + sentiment_value + ' word is within three words to the right of "' + entity.entity + '"'
            else:
                adjective = check_adjectives_before_pos(candidate.text, entity.entity)
                if adjective is not None:
                    sentiment_value = check_sentiment_adjective(adjective)
                    condition = 'A ' + sentiment_value + ' word is within two words to the left of "' + entity.entity + '"'
            if adjective is not None and sentiment_value != 'neutral':
                explanation = Explanation(
                    name='LF_' + str(index),
                    label=label,
                    condition=condition,
                    candidate=candidate,
                )
                explanations.append(explanation)
                index = index + 1
    #            print(str(explanation).upper())

    pos_adjectives_list_ = list(dict.fromkeys(pos_adjectives_list))
    neu_adjectives_list_ = list(dict.fromkeys(neu_adjectives_list))
    neg_adjectives_list_ = list(dict.fromkeys(neg_adjectives_list))

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(pos_adjectives_list_, f)

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(neu_adjectives_list_, f)

    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(neg_adjectives_list_, f)

    exp_io = ExplanationIO()
    exp_io.write(explanations, DATA_FILE6)
    exp_io.read(DATA_FILE6)

    print("Done")


def get_words_before(quantity, sentence, entity):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = sentence.split()
    if entity in words:
        index = words.index(entity)
        before = index - min(index, quantity)
        return ' '.join(map(str, words[before:index]))


def get_words_after(quantity, sentence, entity):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = sentence.split()
    if entity in words:
        index = words.index(entity) + 1
        after = index + min(index, quantity)
        return ' '.join(map(str, words[index:after]))


def check_sentiment_adjective(adjective):
    sid = SentimentIntensityAnalyzer()

    if (sid.polarity_scores(adjective)['compound']) >= 0.1:
        pos_adjectives_list.append(adjective)
        return 'positive'
    elif (sid.polarity_scores(adjective)['compound']) <= -0.1:
        neg_adjectives_list.append(adjective)
        return 'negative'
    else:
        neu_adjectives_list.append(adjective)
        return 'neutral'


def check_adjectives_before_pos(sentence, entity):
    words = get_words_before(2, sentence, entity)
    if words == None or len(words.split(" ")) == 0:
        return None
    else:
        spacy_nlp = spacy.load('en_core_web_sm')
        doc = spacy_nlp(words)
        for token in doc:
            if token.pos_ == 'ADJ':
                return token.text
                break


def check_adjectives_after_verb(sentence, entity):
    words = get_words_after(3, sentence, entity)
    if words is None or len(words.split(" ")) < 3:
        return None
    else:
        spacy_nlp = spacy.load('en_core_web_sm')
        doc = spacy_nlp(words)
        if doc[0].pos_ == 'VERB' and (doc[1].pos_ == 'ADJ' or doc[2].pos_ == 'ADJ'):
            return doc[1].text
