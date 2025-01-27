import json
import os
import urllib

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()
    print(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


file_path = 'instruction-data.json'
url = (
    'https://raw.githubusercontent.com/rasbt/LLMs-from-scratch'
    '/main/ch07/01_main-chapter-code/instruction-data.json'
)

data = download_and_load_file(file_path, url)
#print('number of entries:', len(data))

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
val_portion = len(data) - train_portion - test_portion

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print('training set length', len(train_data))
print('validation set length:', len(val_data))
print('test set length:', len(test_data))