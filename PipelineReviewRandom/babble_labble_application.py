import pickle
from babble.utils import ExplanationIO
from babble import Babbler
from collections import Counter

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE3 = "data/my_explanations.tsv"
DATA_FILE4 = 'data/Ls.pkl'
DATA_FILE5 = 'data/predicted_labels.pkl'
DATA_FILE6 = 'data/perturbations.pkl'


def train():
    print("Start training")

    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    exp_io = ExplanationIO()
    explanations = exp_io.read(DATA_FILE3)

    babbler = Babbler(Cs, Ys,apply_filters=False)

    babbler.apply(explanations, split=0)

    Ls = []
    for split in [0, 1, 2]:
        L = babbler.get_label_matrix(split)
        Ls.append(L)

    L_train = Ls[0].toarray()

    predicted_labels = []

    for line in L_train:
        predicted_labels.append(most_frequent(line))

    perturbations = []

    for right, wrong in zip(Ys[0], predicted_labels):
        perturbations.append(int(right) - wrong)

    perturbations = [abs(perturbation) for perturbation in perturbations]

    len_wrong = 0
    for perturbation in perturbations:
        if perturbation != 0:
            len_wrong = len_wrong + 1
    print('number of wrong are: ' + str(len_wrong))

    with open(DATA_FILE4, 'wb') as f:
        pickle.dump(Ls, f)

    with open(DATA_FILE5, 'wb') as f:
        pickle.dump(predicted_labels, f)

    with open(DATA_FILE6, 'wb') as f:
        pickle.dump(perturbations, f)

    print("Done")


def most_frequent(line):
    filtered_list = list(filter(lambda a: a != 0, line))
    if len(filtered_list) > 0:
        occurence_count = Counter(filtered_list)
        return occurence_count.most_common(1)[0][0]
    else:
        return 0
