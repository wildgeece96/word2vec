import sentencepiece as spm
import numpy as np 
import os 
from tqdm import tqdm 

sp = spm.SentencePieceProcessor()
sp.load("./tokenizer/aozora_8k_model.model")

vocab_size = sp.get_piece_size()
x_hist = np.zeros(vocab_size, dtype=np.float64)  
with open("./data/corpus.txt", "r") as f:
    corpus = f.readlines()
token_num = 0
for line in tqdm(corpus): 
    ids = sp.EncodeAsIds(line)
    token_num += len(ids)
    for id in ids:
        x_hist[id] += 1   

x_hist /= x_hist.sum()
x_dist = x_hist*0.95 + 0.05/8000
os.makedirs("./out", exist_ok=True)
np.save("./out/x_dist.npy", x_hist)
print("token_num :", token_num)
