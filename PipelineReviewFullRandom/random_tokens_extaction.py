import pickle
import random
import spacy
import progressbar


DATA_FILE1 = 'data/train_examples.pkl'
DATA_FILE2 = 'data/train_labels.pkl'


def extract_token(iteration_number):
    print("Extracting tokens")

    DATA_FILE3 = 'data/tokens/correct_tokens_list' + str(iteration_number - 1) + '.pkl'
    DATA_FILE4 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'

    with open(DATA_FILE1, 'rb') as f:
        examples = pickle.load(f)
    with open(DATA_FILE2, 'rb') as f:
        labels = pickle.load(f)
    try:
        with open(DATA_FILE3, 'rb') as f:
            correct_tokens_list = pickle.load(f)
    except:
        correct_tokens_list = [[] for i in range(len(examples))]

    tokens_train_list = []


    spacy_nlp = spacy.load('en_core_web_sm')

    for example, label, correct_tokens in progressbar.progressbar(zip(examples, labels, correct_tokens_list)):
        doc = spacy_nlp(example)
        tokenized_sentence = [token.text for token in doc if not token.is_stop and token.is_alpha]

        selected_words = select_random_words(tokenized_sentence, label, correct_tokens)
        tokens_train_list.append(selected_words)

    print(len(tokens_train_list))
    print(correct_tokens_list[0])
    print(tokens_train_list[0])

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(tokens_train_list, f)

    with open(DATA_FILE3, 'wb') as f:
        pickle.dump(correct_tokens_list, f)

    print("Done")


def select_random_words(tokens, label, correct_tokens):
    selected_words = []
    wrong_indexes = []
    choiced_indexes = []
    perturbation = 4 - len(correct_tokens)
    while not (perturbation == 0):
        index = random.sample(range(len(tokens)), 1)[0]
        if index not in wrong_indexes and index not in choiced_indexes:
            new = (tokens[index], label)
            if new not in correct_tokens:
                perturbation = perturbation - 1
                selected_words.append(new)
                choiced_indexes.append(index)
            else:
                wrong_indexes.append(index)
    return selected_words + correct_tokens
