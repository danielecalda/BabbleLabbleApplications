from src.PipelineReviewRandomTokens.setup import setup
from src.PipelineReviewRandomTokens.random_nouns_expressions_extraction import extract_nouns, extract_expressions
from src.PipelineReviewRandomTokens.write_explanations import write_explanations_for_expressions
from src.PipelineReviewRandomTokens.babble_labble_application import train_for_expressions
import os


filelist = ['data/train_examples.pkl', 'data/dev_examples.pkl', 'data/test_examples.pkl', 'data/train_labels.pkl'
                , 'data/dev_labels.pkl', 'data/test_labels.pkl', 'data/data.pkl', 'data/labels.pkl']
if not all([os.path.isfile(f) for f in filelist]):
    setup()

for i in range(1, 2):
    extract_nouns(i)
    extract_expressions(i)
    write_explanations_for_expressions(i)
    train_for_expressions(i)
