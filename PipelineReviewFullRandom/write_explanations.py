import pickle
from babble import Explanation
from babble.utils import ExplanationIO
import progressbar

DATA_FILE1 = 'data/train_labels.pkl'


def write_explanations(iteration_number):
    print("Writing explanations")

    DATA_FILE2 = 'data/tokens/tokens_train_list' + str(iteration_number) + '.pkl'
    DATA_FILE3 = 'data/explanations/my_explanations' + str(iteration_number) + '.tsv'

    with open(DATA_FILE1, 'rb') as f:
        labels = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        tokens_list = pickle.load(f)

    index = 0
    explanations = []

    for label, selected_words in progressbar.progressbar(zip(labels, tokens_list)):

        for word in selected_words:
            explanation = Explanation(
                name='LF_' + str(index),
                label=int(label),
                condition=create_condition(word),
            )

            explanations.append(explanation)
            index = index + 1

    exp_io = ExplanationIO()
    exp_io.write(explanations, DATA_FILE3)
    exp_io.read(DATA_FILE3)

    print("Done")


def create_condition(word):
    condition = 'the word ' + '"' + word + '" is in the sentence'
    return condition
