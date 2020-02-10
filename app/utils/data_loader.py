import sentencepiece as spm
import random
import numpy as np 
import tensorflow as tf 

class DataLoader(object):

    def __init__(self, window_size, neg_sample_num, 
                    corpus_path="./data/corpus.txt", 
                    sp_path="./tokenizer/aozora_8k_model.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)
        self.corpus_path = corpus_path 
        self.window_size = window_size
        self.neg_sample_num = neg_sample_num
        self.vocab_size = self.sp.get_piece_size()  

    def load(self, x_dist, batch_size=100, shuffle=True):
        with open(self.corpus_path, "r") as f:
            lines = f.readlines()
        batch = []
        y = [] 
        while True:
            if shuffle:
                random.shuffle(lines)  
            for line in lines:   
                ids = self.sp.EncodeAsIds(line) 
                if len(ids) < self.window_size*2+1:
                    continue
                _batch, _y = self.make_batch(ids) 
                batch += _batch
                y += _y 
                while len(batch) >= batch_size:
                    negative_samples = np.random.choice(range(self.vocab_size),
                                    size=(batch_size,self.neg_sample_num), 
                                    p=x_dist)
                    negative_samples[:, 0] = y[:batch_size]
                    yield (tf.convert_to_tensor(batch[:batch_size], dtype=tf.int32),
                                tf.convert_to_tensor(negative_samples, dtype=tf.int32))
                    batch = batch[batch_size:] 
                    y = y[batch_size:]

    def make_batch(self, ids):
        w_size = self.window_size 
        mini_batch = []
        y = [] 
        for i in range(w_size, len(ids)-w_size):
            _ids =  ids[i-w_size:i] + ids[i+1:i+w_size+1]
            mini_batch.append(_ids)
            y.append(ids[i])
        return mini_batch, y

        
