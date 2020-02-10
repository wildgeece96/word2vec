import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm 
import numpy as np 
import datetime 


class Trainer(object):
    def __init__(self, model, loader, x_dist, optimizer): 
        self.model = model 
        self.bce = keras.losses.BinaryCrossentropy()
        self.loader = loader
        self.neg_sample_num = self.loader.neg_sample_num
        self.x_dist = x_dist
        self.optimizer = optimizer
    
    def train(self, batch_size, epochs=10):
        self.batch_size = batch_size 
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.writer = tf.summary.create_file_writer(f"./out/record/{current_time}")
        for epoch in range(epochs):
            self.train_epoch(epoch, epochs)
    
    def train_epoch(self, epoch, epochs):
        with tqdm(self.loader.load(self.x_dist, self.batch_size)) as pbar:
            pbar.set_description(f"[Epoch {epoch:02d}/{epochs:02d}]")
            for i, (batch, negative_samples) in enumerate(pbar): 
                loss_value = self.train_step(batch, negative_samples)
                with self.writer.as_default(): 
                    tf.summary.scalar("loss", loss_value.numpy(), step=i)
                    self.writer.flush()
                pbar.set_postfix({"loss" : loss_value.numpy(), "samples" : i*self.batch_size})

    def train_step(self, inputs, negative_samples):
        with tf.GradientTape() as tape:
                loss_value = self.loss(inputs, negative_samples)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value
        
    def loss(self, inputs, negative_samples):
        logits = self.model(inputs, negative_samples)
        y = np.zeros([self.batch_size, logits.shape[-1]], dtype=np.float32)
        y[:, 0] =  1
        y = tf.convert_to_tensor(y, dtype='float32')
        loss = self.bce(logits, y)
        return loss 

    def save_model(self, save_path):
        model.save(save_path) 
