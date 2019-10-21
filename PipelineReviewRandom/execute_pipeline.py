from src.PipelineReviewRandom.setup import setup
from src.PipelineReviewRandom.random_tokens_extaction import extract_token
from src.PipelineReviewRandom.write_explanations import write_explanations
from src.PipelineReviewRandom.babble_labble_application import train

setup()
extract_token()
write_explanations()
train()
