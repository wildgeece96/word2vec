import sys
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt

# Evaluate the vectors.
vec_path = sys.argv[1]
# e.g. ./out/record/20200211-1011/wordvec_009.npy
sp_path = "./tokenizer/aozora_8k_model.model"

sp = spm.SentencePieceProcessor()
sp.load(sp_path)
word_vec = np.load(vec_path)

print("Enter the word")
string = input()

ids = sp.EncodeAsIds(string)


def find_similar(id, word_vec, topn=5):
    query_vec = word_vec[id][:, np.newaxis]
    # normalize vectors
    word_vec /= np.sqrt(np.power(word_vec, 2).sum(axis=1, keepdims=True))
    cosine = np.dot(word_vec, query_vec).flatten()
    # most similar word is the same word for id.
    most_similar_ids = np.argsort(cosine)[-2-topn:-1]
    print("Original word : ", sp.DecodeIds([id]))
    for i in range(topn):
        print(f"\tTop {i:02d} : ", sp.DecodeIds([int(most_similar_ids[-2-i])]))


for id in ids:
    find_similar(id, word_vec, topn=5)
