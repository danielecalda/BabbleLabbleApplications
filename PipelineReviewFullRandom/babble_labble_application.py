import pickle
from babble.utils import ExplanationIO2
from babble import Babbler
from collections import Counter

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'


def train(iteration_number):
    DATA_FILE3 = 'data/explanations/my_explanations' + str(iteration_number) + '.tsv'
    DATA_FILE4 = 'data/Ls/Ls' + str(iteration_number) + '.pkl'
    DATA_FILE5 = 'data/results/predicted_training_labels'  + str(iteration_number) + '.pkl'
    DATA_FILE6 = 'data/tokens/correct_tokens_list' + str(iteration_number) + '.pkl'
    DATA_FILE7 = 'data/results/summary' + str(iteration_number) + '.txt'
    DATA_FILE8 = 'data/results/predicted_test_labels' + str(iteration_number) + '.pkl'
    DATA_FILE9 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'

    print("Start training")

    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE9, 'rb') as f:
        tokens_train_list = pickle.load(f)

    exp_io = ExplanationIO2()
    explanations = exp_io.read(DATA_FILE3)

    babbler = Babbler(Cs, Ys)

    babbler.apply(explanations, split=0)

    parses = babbler.get_parses(translate=False)

    token_from_parses = create_tokens_from_parses(parses)

    Ls = []
    for split in [0, 1, 2]:
        L = babbler.get_label_matrix(split)
        Ls.append(L)

    L_train = Ls[0].toarray()
    L_test = Ls[2].toarray()

    predicted_training_labels = []
    predicted_test_labels = []

    for line in L_train:
        predicted_training_labels.append(most_frequent(line))

    for line in L_test:
        predicted_test_labels.append(most_frequent(line))

    len_wrong_train = 0
    for right, wrong in zip(Ys[0], predicted_training_labels):
        if int(right) != wrong:
            len_wrong_train += 1

    correct_tokens_list = []

    for tokens_list in tokens_train_list:
        correct_tokens = []
        for token in tokens_list:
            if token in token_from_parses:
                correct_tokens.append(token)
        correct_tokens_list.append(correct_tokens)


    len_wrong_test = 0
    for right, wrong in zip(Ys[2], predicted_test_labels):
        if int(right) != wrong:
            len_wrong_test += 1

    training_accuracy = percentage(len(Ys[0]) - len_wrong_train, len(Ys[0]))

    test_accuracy = percentage(len(Ys[2]) - len_wrong_test, len(Ys[2]))

    print("Number of wrong in training set: " + str(len_wrong_train))
    print("Number of wrong in test set: " + str(len_wrong_test))
    print("Training Accuracy: " + str(training_accuracy))
    print("Test Accuracy " + str(test_accuracy))

    with open(DATA_FILE7, 'a') as f:
        f.write("Iteration number: " + str(iteration_number))
        f.write("Number of wrong in training set: " + str(len_wrong_train))
        f.write('\n')
        f.write("Number of wrong in test set: " + str(len_wrong_test))
        f.write('\n')
        f.write("Training Accuracy: " + str(training_accuracy))
        f.write('\n')
        f.write("Test Accuracy: " + str(test_accuracy))

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(Ls, f)

    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(predicted_training_labels, f)

    with open(DATA_FILE8, 'wb') as f:
        pickle.dump(predicted_test_labels, f)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(correct_tokens_list, f)

    print("Done")


def most_frequent(line):
    filtered_list = list(filter(lambda a: a != 0, line))
    if len(filtered_list) > 0:
        occurence_count = Counter(filtered_list)
        return occurence_count.most_common(1)[0][0]
    else:
        return 0


def percentage(part, whole):
    return 100 * float(part)/float(whole)

def create_tokens_from_parses(parses):
    tokens_from_parses = []
    for parse in parses:
        explanation = parse.explanation
        word = explanation.word
        label = explanation.label
        new = (word, label)
        tokens_from_parses.append(new)
    return tokens_from_parses


