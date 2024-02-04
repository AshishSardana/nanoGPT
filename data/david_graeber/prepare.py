import os
import requests
import tiktoken
import numpy as np

# path to .txt directory
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
input_directory = os.path.join(os.path.dirname(__file__), '../../../dg_talk/books/txt/')
output_directory = os.path.join(os.path.dirname(__file__), '../../../dg_talk/books/bin/')

data = ''
for file_name in os.listdir(input_directory):
    file_path = os.path.join(input_directory, file_name)

    with open(file_path, 'r') as f:
        content = f.read()
        data += content

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_directory, 'train.bin'))
val_ids.tofile(os.path.join(output_directory, 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
