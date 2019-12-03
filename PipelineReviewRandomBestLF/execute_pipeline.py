from src.PipelineReviewRandomTokens.setup import setup
from src.PipelineReviewRandomTokens.random_tokens_extaction import extract_token
from src.PipelineReviewRandomTokens.write_explanations import write_explanations
from src.PipelineReviewRandomTokens.babble_labble_application import train
import os


filelist = ['data/train_examples.pkl', 'data/dev_examples.pkl', 'data/test_examples.pkl', 'data/train_labels.pkl'
                , 'data/dev_labels.pkl', 'data/test_labels.pkl', 'data/data.pkl', 'data/labels.pkl']
if not all([os.path.isfile(f) for f in filelist]):
    setup()

for i in range(1, 10):
    extract_token(i)
    write_explanations(i)
    train(i)
