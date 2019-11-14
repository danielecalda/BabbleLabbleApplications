import pickle
import progressbar
from babble import Babbler
from babble.utils import ExplanationIO2

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/labels.pkl'
DATA_FILE3 = 'data/explanations/my_explanations_expressions1.tsv'



def test2():
    with open(DATA_FILE1, 'rb') as f:
        Cs = pickle.load(f)

    with open(DATA_FILE2, 'rb') as f:
        Ys = pickle.load(f)

    exp_io = ExplanationIO2()
    explanations = exp_io.read(DATA_FILE3)

    babbler = Babbler(Cs, Ys)

    babbler.apply(explanations, split=0)

    Ls = []
    for split in [0, 1, 2]:
        L = babbler.get_label_matrix(split)
        Ls.append(L)

    babbler.commit()

    parses = babbler.get_parses(translate=False)
    '''
    parse = parses[222]
    print(parse.explanation)
    
    for candidate,label in zip(Cs[0],Ys[0]):
        prediction = parse.function(candidate)
        if prediction != 0:
            print('prediction ' + str(prediction))
            print('label ' + str(label))
            print(candidate.text)
    '''
    index = 201
    candidate = Cs[0][index]
    print(candidate.text)
    print('\n')
    prediction = Ys[0][index]
    print('prediction ' + str(prediction))
    print('\n')

    labels = []
    explanations = []
    for parse in parses:
        label = parse.function(candidate)
        if label != 0:
            explanations.append(parse.explanation)
            labels.append(label)

    print(explanations)
    print(labels)




test2()