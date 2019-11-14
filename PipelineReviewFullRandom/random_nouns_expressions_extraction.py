import progressbar
import pickle
import random
import spacy
from src.PipelineReviewFullRandom.utils import check_adjective_noun, check_adjective_after_verb

DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/train_labels.pkl'

def extract_nouns(iteration_number):
    DATA_FILE3 = 'data/expressions/nouns_list' + str(iteration_number) + '.pkl'

    with open(DATA_FILE1, 'rb') as f:
        examples = pickle.load(f)

    spacy_nlp = spacy.load('en_core_web_sm')

    try:
        with open(DATA_FILE3, 'rb') as f:
            nouns_list = pickle.load(f)
    except:
        nouns_list = []

    for example in progressbar.progressbar(examples):
        doc = spacy_nlp(example)

        nouns = [token.text for token in doc if not token.is_stop and token.is_alpha and token.tag_ in 'NN'
                 and not token.ent_type_ in ('PERSON', 'GPE', 'NORP', 'DATE', 'CARDINAL','LOC')]

        nouns_list.append(nouns)

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(nouns_list, f)


def extract_expressions(iteration_number):
    DATA_FILE3 = 'data/expressions/nouns_list' + str(iteration_number) + '.pkl'
    DATA_FILE4 = 'data/expressions/expressions_list' + str(iteration_number) + '.pkl'

    with open(DATA_FILE3, 'rb') as f:
        nouns_list = pickle.load(f)

    with open(DATA_FILE1, 'rb') as f:
        train_examples = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        train_labels = pickle.load(f)

    expressions_list = []

    print(len(train_examples))
    print(len(nouns_list))

    for example, label, nouns in progressbar.progressbar(zip(train_examples, train_labels, nouns_list)):

        expressions = []

        if len(nouns) >= 3:
            indexes = random.sample(range(len(nouns)), 3)
            for index in indexes:
                adjective = check_adjective_after_verb(example, nouns[index])
                if adjective is not None:
                    expressions.append((nouns[index] + ' ' + adjective, label))
                else:
                    adjective = check_adjective_noun(example, nouns[index])
                    if adjective is not None:
                        expressions.append((adjective + ' ' + nouns[index], label))
        expressions_list.append(expressions)

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(expressions_list, f)
