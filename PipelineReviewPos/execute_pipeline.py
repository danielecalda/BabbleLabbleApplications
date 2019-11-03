from src.PipelineReviewPos.setup import setup
from src.PipelineReviewPos.write_explanations import write_explanations
from src.PipelineReviewPos.babble_labble_application import train


setup()
write_explanations()
train()