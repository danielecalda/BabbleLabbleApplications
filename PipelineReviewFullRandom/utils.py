import pickle
from babble import Babbler
from babble.utils import ExplanationIO2

DATA_FILE1 = 'data/data.pkl'
DATA_FILE2 = 'data/tokens/correct_tokens_list1.pkl'

with open(DATA_FILE1, 'rb') as f:
    Cs = pickle.load(f)

with open(DATA_FILE2, 'rb') as f:
    correct_tokens_list = pickle.load(f)

print(correct_tokens_list)

