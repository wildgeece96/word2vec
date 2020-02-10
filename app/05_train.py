import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 

from utils.trainer import Trainer
from utils.modeling import CBOW
from utils.data_loader import DataLoader 

window_size = 5
num_neg_samples = 20
hidden_dim = 100
batch_size = 100
epochs = 10
corpus_path = "./data/corpus.txt"
sp_path = "./tokenizer/aozora_8k_model.model"
x_dist = np.load("./out/x_dist.npy")

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

loader = DataLoader(window_size, num_neg_samples, corpus_path=corpus_path,
                sp_path=sp_path)
vocab_size = loader.vocab_size
model = CBOW(hidden_dim, vocab_size, window_size, num_neg_samples=num_neg_samples)
trainer = Trainer(model, loader, x_dist, optimizer)
trainer.train(batch_size, epochs=epochs)
trainer.save_model("./out/model.h5")
print("Model has been saved.")
