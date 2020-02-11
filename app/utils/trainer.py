import os
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm
import numpy as np
import datetime


class Trainer(object):
    def __init__(self, model, loader, x_dist, optimizer): 
        self.model = model
        self.bce = tf.nn.softmax_cross_entropy_with_logits
        self.loader = loader
        self.neg_sample_num = self.loader.neg_sample_num
        self.x_dist = x_dist
        self.optimizer = optimizer
        self.root = tf.train.Checkpoint(optimizer=self.optimizer,
                            model=self.model)
        self.step = 0
        self.time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = f"./out/record/{self.time}"
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.checkpoint_dir = os.path.join(self.log_dir, "model_dir")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, batch_size, epochs=10):
        self.batch_size = batch_size
        for epoch in range(epochs):
            self.train_epoch(epoch, epochs)
            self.save_model()
            self.save_wordvec(epoch)
    
    def train_epoch(self, epoch, epochs):
        with tqdm(self.loader.load(self.x_dist, self.batch_size)) as pbar:
            pbar.set_description(f"[Epoch {epoch:02d}/{epochs:02d}]")
            for i, (batch, ys) in enumerate(pbar): 
                loss_value, _ = self.train_step(batch, ys)
                self.step += 1 
                with self.writer.as_default(): 
                    tf.summary.scalar("loss", loss_value.numpy(), step=self.step)
                    self.writer.flush()
                pbar.set_postfix({"loss" : loss_value.numpy(), "samples" : i*self.batch_size})

    def train_step(self, inputs, ys):
        with tf.GradientTape() as tape:
            loss_value, log_softmax = self.loss(inputs, ys)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value, log_softmax
        
    def loss(self, inputs, ys):
        log_softmax = self.model(inputs)
        y = np.zeros([self.batch_size, log_softmax.shape[-1]], dtype=np.float32)
        for i, _idx in enumerate(ys): 
            y[i, _idx] =  1
        y = tf.convert_to_tensor(y, dtype='float32')
        loss = tf.reduce_sum(- y * log_softmax)/self.batch_size
        return loss, log_softmax

    def save_model(self):
        prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.root.save(prefix)

    def load_model(self, checkpoint_dir):
        self.root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def save_wordvec(self, epoch):
        weights = self.model.embedding.embedding.get_weights()
        weight = weights[0]
        path = os.path.join(self.log_dir, f"wordvec_{epoch:03d}.npy")
        np.save(path, weight)

