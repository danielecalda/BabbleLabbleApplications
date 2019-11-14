from src.PipelineReviewFullRandom.setup import setup
from src.PipelineReviewFullRandom.random_nouns_expressions_extraction import extract_nouns, extract_expressions
from src.PipelineReviewFullRandom.write_explanations import write_explanations_for_expressions
from src.PipelineReviewFullRandom.babble_labble_application import train_for_expressions
import os


filelist = ['data/train_examples.pkl', 'data/dev_examples.pkl', 'data/test_examples.pkl', 'data/train_labels.pkl'
                , 'data/dev_labels.pkl', 'data/test_labels.pkl', 'data/data.pkl', 'data/labels.pkl']
if not all([os.path.isfile(f) for f in filelist]):
    setup()
'''
for i in range(1, 20):
    extract_token(i)
    write_explanations_for_tokens(i)
    train_for_tokens(i)

'''
extract_nouns(1)
extract_expressions(1)
extract_expressions(1)
write_explanations_for_expressions(1)
train_for_expressions(1)
