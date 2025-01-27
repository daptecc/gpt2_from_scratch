import time
import torch
from tqdm import tqdm
import re

from instruction_dataset import format_input
from instruction_dataloader import *
from instruction_loadmodel import *
from train import *

from plot import plot_values

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-5, weight_decay=0.1
)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f'Training completed in {execution_time_minutes:.2f} minutes.')

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_values(epochs_tensor, tokens_seen, train_losses,
            val_losses, fig_name='instruction', label='loss')

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG['context_length'],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace('### Response:', '')
        .strip()
    )
    test_data[i]['model_response'] = response_text

with open('instruction-data-with-response.json', 'w') as file:
    json.dump(test_data, file, indent=4)

file_name = f"{re.sub(r' ()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), file_name)
print(f'Model saved as {file_name}')