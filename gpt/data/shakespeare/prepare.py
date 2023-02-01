import os
import requests
import tiktoken
import numpy as np

# Get data in input.txt
DIR_NAME = os.path.dirname(__file__)
data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
with open(os.path.join(DIR_NAME, "input.txt"), "w") as f:
    f.write(requests.get(data_url).text)

# Split train val data with 90% split
TRAIN_VAL_SPLIT = 0.9
with open(os.path.join(DIR_NAME, "input.txt"), "r") as f:
    input_txt = f.read()
    print("number of strings:", len(input_txt))
    train_data = input_txt[0:int(TRAIN_VAL_SPLIT * len(input_txt))]
    val_data = input_txt[int(TRAIN_VAL_SPLIT * len(input_txt)):]

# encode with tiktoken
enc = tiktoken.get_encoding("gpt2")
train_tokens = enc.encode(train_data)
val_tokens = enc.encode(val_data)
assert enc.decode(train_tokens) == train_data
assert enc.decode(val_tokens) == val_data

# Output train val to binary file
output_train_bin_file = os.path.join(DIR_NAME, "train.bin")
output_val_bin_file = os.path.join(DIR_NAME, "val.bin")
np.array(train_tokens).astype(np.uint16).tofile(output_train_bin_file)
np.array(val_tokens).astype(np.uint16).tofile(output_val_bin_file)
print("Number of train tokens", len(train_tokens))
print("Number of val tokens", len(val_tokens))

# Train has 301966 tokens
# Val has 36059 tokens