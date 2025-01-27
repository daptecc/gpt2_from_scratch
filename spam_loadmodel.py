import torch
from config import BASE_CONFIG, model_configs
from gptmodel import GPTModel
from gpt_download import download_and_load_gpt2
from load_weights_into_gpt import load_weights_into_gpt2
from generate_text import generate_text_simple, text_to_token_ids, token_ids_to_text

import tiktoken

CHOOSE_MODEL = 'gpt2-small (124M)'
INPUT_PROMPT = 'Every effort moves'
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

tokenizer = tiktoken.get_encoding('gpt2')

model_size = CHOOSE_MODEL.split(' ')[-1].lstrip('(').rstrip(')')
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir='gpt2'
)


model = GPTModel(BASE_CONFIG)
load_weights_into_gpt2(model, params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
torch.manual_seed(123)

text_1 = 'every effort moves you'
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer).to(device),
    max_new_tokens=15,
    context_size=BASE_CONFIG['context_length']
)
print(token_ids_to_text(token_ids, tokenizer))

text_2 = (
    "is the following text 'spam'? Answers with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
print(token_ids_to_text(token_ids, tokenizer))