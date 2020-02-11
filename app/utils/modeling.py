import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    LayerNormalization
)


class ContextEmbeddingLayer(keras.layers.Layer):

    def __init__(self, vocab_size, hidden_dim, input_length=10, **kwargs):
        super(ContextEmbeddingLayer, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, hidden_dim, 
                input_length=input_length, name="Embedding")
        self.bias = self.add_weight(shape=hidden_dim, dtype=tf.float32, 
                            initializer='zero', name="Embedding_bias")
        self.norm = LayerNormalization(axis=-2, name='norm')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = tf.reduce_mean(x, axis=1) + self.bias
        x = self.norm(x)
        return x


class NegativeSamplingLayer(keras.layers.Layer):
    
    def __init__(self, hidden_dim, vocab_size, num_sample=5): 
        super(NegativeSamplingLayer, self).__init__()
        self.out_embedding = Embedding(vocab_size, hidden_dim, input_length=5,
                        name="out_embedding")
        self.num_sample = num_sample

    def call(self, inputs, idxs):
        '''
        inputs : (batch_size, hidden_size).
        idxs : (batch_size, num_sample). 
        '''
        inputs = inputs[:, tf.newaxis, :]
        negative_embed = self.out_embedding(idxs) 
        # (batch_size, num_sample, hidden_dim)
        negative_embed = tf.transpose(negative_embed, perm=[0, 2, 1])
        # (batch_size, hidden_dim, num_sample)
        x = tf.matmul(inputs, negative_embed)
        x = tf.math.sigmoid(x)
        x = tf.squeeze(x, axis=1)
        return x

class CBOW(keras.Model):

    def __init__(self, hidden_dim=100, vocab_size=8000, window_size=5, num_neg_samples=10):
        super(CBOW, self).__init__()
        self.embedding = ContextEmbeddingLayer(vocab_size, hidden_dim, window_size*2, 
                        name='context_Embedding_layer')
        self.decode = Dense(vocab_size, input_shape=(hidden_dim,), name="decode")
    
    def call(self, inputs): 
        """
        inptus : tf.Tensor. (batch_size, window_size*2).
        negative_samples : tf.Tensor. (batch_size, num_neg_samples). 
        """
        x = self.embedding(inputs)
        x = self.decode(x)
        x = tf.nn.relu(x)
        x = tf.nn.log_softmax(x, axis=-1)
        return x



        