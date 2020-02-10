import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.layers import (
    Embedding
)

class ContextEmbeddingLayer(keras.layers.Layer):

    def __init__(self, vocab_size, hidden_dim, input_length=10):
        super(ContextEmbeddingLayer, self).__init__()
        self.embedding = Embedding(vocab_size, hidden_dim, input_length=input_length)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1) 
        return x

class NegativeSamplingLayer(keras.layers.Layer):
    
    def __init__(self, hidden_dim, vocab_size, num_sample=5): 
        super(NegativeSamplingLayer, self).__init__()
        self.out_embedding = Embedding(vocab_size, hidden_dim, input_length=5)
        self.num_sample = num_sample 

    def call(self, inputs, idxs):
        '''
        inputs : (batch_size, hidden_size).
        idxs : (batch_size, num_sample). 
        '''
        negative_embed = self.out_embedding(idxs) 
        # (batch_size, num_sample, hidden_dim)
        negative_embed = tf.transpose(negative_embed, perm=[0, 2, 1])
        # (batch_size, hidden_dim, num_sample)
        x = tf.matmul(inputs, negative_embed)
        x = tf.math.sigmoid(x)
        return x

class CBOW(keras.layers.Layer):

    def __init__(self, hidden_dim=100, vocab_size=8000, window_size=5, num_neg_samples=10):
        super(CBOW, self).__init__()
        self.embedding = ContextEmbeddingLayer(vocab_size, hidden_dim, window_size*2)
        self.negatvie_sampling_dot = NegativeSamplingLayer(hidden_dim, vocab_size, num_sample=num_neg_samples)
    
    def call(self, inputs, negative_samples): 
        """
        inptus : tf.Tensor. (batch_size, window_size*2).
        negative_samples : tf.Tensor. (batch_size, num_neg_samples). 
        """
        x = self.embedding(inputs)
        x = self.negatvie_sampling_dot(x, negative_samples)
        x = tf.math.sigmoid(x)
        return x



        