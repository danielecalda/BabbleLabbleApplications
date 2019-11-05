from src.PipelineReviewFullRandom.setup import setup
from src.PipelineReviewFullRandom.random_tokens_extaction import extract_token
from src.PipelineReviewFullRandom.write_explanations import write_explanations
from src.PipelineReviewFullRandom.babble_labble_application import train
import os


filelist = ['data/train_examples.pkl', 'data/dev_examples.pkl', 'data/test_examples.pkl', 'data/train_labels.pkl'
                , 'data/dev_labels.pkl', 'data/test_labels.pkl', 'data/data.pkl', 'data/labels.pkl']
if not all([os.path.isfile(f) for f in filelist]):
    setup()
'''
for i in range(1, 10):
    extract_token(i)
    write_explanations(i)
    train(i)
'''

extract_token(1)
write_explanations(1)
train(1)
