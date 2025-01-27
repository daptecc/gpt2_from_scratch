from gpt_download import download_and_load_gpt2
from gptmodel import GPTModel
from load_weights_into_gpt import load_weights_into_gpt

from config import *

CHOOSE_MODEL = 'gpt2_medium (355M)'
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split()[-1].lstrip('(').rstrip(')')

settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir='gpt2'
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()