import pickle
from babble import Explanation
import progressbar
from babble import Babbler
from src.PipelineReviewFullRandom.utils import most_frequent
from src.PipelineReviewFullRandom.utils import calculate_number_wrong
from src.PipelineReviewFullRandom.utils import percentage

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE3 = 'data/tokens/correct_tokens_list7.pkl'
DATA_FILE7 = 'data/results/summary.txt'


def apply_results():
    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    with open(DATA_FILE3, 'rb') as f:
        correct_tokens_list = pickle.load(f)

    tokens = []
    for correct_tokens in correct_tokens_list:
        for token in correct_tokens:
            if token not in tokens:
                tokens.append(token)

    index = 0
    explanations = []

    for word in progressbar.progressbar(tokens):
        explanation = Explanation(
            name='LF_' + str(index),
            label=word[1],
            condition=create_condition(word[0]),
            word=word[0]
        )

        explanations.append(explanation)
        index = index + 1

    print(explanations)

    babbler = Babbler(Cs, Ys)

    babbler.apply(explanations, split=0)

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
    count = 0
    for label in predicted_training_labels:
        if label != 0:
            count += 1
    print(count)

    for line in L_test:
        predicted_test_labels.append(most_frequent(line))

    len_wrong_train = calculate_number_wrong(Ys[0], predicted_training_labels)

    len_wrong_test = calculate_number_wrong(Ys[2], predicted_test_labels)

    training_accuracy = percentage(len(Ys[0]) - len_wrong_train, len(Ys[0]))

    test_accuracy = percentage(len(Ys[2]) - len_wrong_test, len(Ys[2]))

    with open(DATA_FILE7, 'a') as f:
        f.write("Iteration number: ")
        f.write('\n')
        f.write("Number of wrong in training set: " + str(len_wrong_train))
        f.write('\n')
        f.write("Number of wrong in test set: " + str(len_wrong_test))
        f.write('\n')
        f.write("Training Accuracy: " + str(training_accuracy))
        f.write('\n')
        f.write("Test Accuracy: " + str(test_accuracy))
        f.write('\n')
        f.write('\n')


def create_condition(word):
    condition = 'the word ' + '"' + word + '" is in the sentence'
    return condition

apply_results()