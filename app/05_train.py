import tensorflow as tf
import numpy as np

from utils.trainer import Trainer
from utils.modeling import CBOW
from utils.data_loader import DataLoader

window_size = 5
num_neg_samples = 100
hidden_dim = 100
batch_size = 1000
epochs = 20
corpus_path = "./data/corpus.txt"
sp_path = "./tokenizer/aozora_8k_model.model"
x_dist = np.load("./out/x_dist.npy")

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# optimizer = tf.keras.optimizers.Adadelta(learning_rate=1e-2)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5)
loader = DataLoader(window_size, num_neg_samples, corpus_path=corpus_path,
                                                    sp_path=sp_path)
vocab_size = loader.vocab_size
model = CBOW(hidden_dim, vocab_size, window_size)
model.compile(optimizer='adam', loss='categorical_crossentropy')
trainer = Trainer(model, loader, x_dist, optimizer)
trainer.train(batch_size, epochs=epochs//2)
trainer.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
trainer.train(batch_size, epochs=epochs//2)
trainer.save_model()
print("Model has been saved.")
