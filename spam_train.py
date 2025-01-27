import torch
import time
import tiktoken

from config import BASE_CONFIG, model_configs
from gptmodel import GPTModel
from gpt_download import download_and_load_gpt2
from load_weights_into_gpt import load_weights_into_gpt2
from spam_dataloader import *
from plot import plot_values

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f'Ep {epoch+1} (Step {global_step:06d}): '
                      f'Train loss {train_loss:.3f}, '
                      f'Val loss {val_loss:.3f}'
                      )

        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f'Training accuracy: {train_accuracy*100:.2f}% | ', end='')
        print(f'Validation accuracy: {val_accuracy*100:.2f}%')
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def classify_review(
        text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    model.val()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return 'spam' if predicted_label == 1 else 'not spam'



if __name__ == '__main__':
    start_time = time.time()
    torch.manual_seed(123)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=50,
            eval_iter=5
        )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f'Training completed in {execution_time_minutes:.2f} minutes.')

    # plot losses
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # plot accuracies
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs,
                val_accs, fig_name='spam', label='accuracy')

    # accuracy metrics
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f'Train accuracy: {train_accuracy*100:.2f}%')
    print(f'Validation accuracy: {val_accuracy*100:.2f}%')
    print(f'Test accuracy: {test_accuracy*100:.2f}%')

    model_path = 'spam_classifier.pth'
    print(f'Saving model to {model_path}')
    torch.save(model.state_dict(), model_path)