
import sys, os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def my_gru(units):
        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        # the code automatically does that.
        if tf.test.is_gpu_available():
            print('### This is CuDNNGRU ###')
            return tf.keras.layers.CuDNNGRU(units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
        else:
            return tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       recurrent_activation='sigmoid')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = my_gru(self.hidden_dim)
    def call(self, enc_inputs, hidden_state):
        x = self.embedding(enc_inputs)
        output, state = self.gru(x, initial_state=hidden_state)
        return output, state
    def initial_hidden_state(self, batch_size):
        # batch_size is passed by the input shape in case the last batch is not exact the samp as self.batch_size
        return tf.zeros((batch_size, self.hidden_dim))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = my_gru(hidden_dim)
        self.final_probs = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.hidden_dim)
        self.W2 = tf.keras.layers.Dense(self.hidden_dim)
        self.V = tf.keras.layers.Dense(1)
    def call(self, dec_input, hidden_state, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_dim)

        # hidden_state shape == (batch_size, hidden_dim)
        # hidden_with_time_axis == (batch_size, 1, hidden_dim)
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)

        # addition score. score shape == (batch_size, max_length, 1)
        # This also can be changed to dot score
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention weights shape == (batch_size, max_length, 1)
        # for each input sequence and each word in the sequence, there is a corresponding weight
        attention_weights = tf.nn.softmax(score, axis=1)

        # add all encoder_output with multiplying their weights
        # shape == (batch_size, max_length, hidden_dim)
        context_vector = attention_weights * enc_output
        # shape == (batch_size, hidden_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x is only one word of decoder. shape == (batch_size, 1, embedding_dim)
        x = self.embedding(dec_input)

        # concat attention vector and decoder input
        # shape == (batch_size, 1, hidden_dim + embedding_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # output shape == (batch_size, 1, hidden_dim)
        output, state = self.gru(x, initial_state=hidden_state)
        # shape == (batch * 1, hidden_dim)
        output = tf.reshape(output, (-1, output.shape[2]))

        # shape == (batch_size * 1, vocab_size)
        probs = self.final_probs(output)

        return probs, state, attention_weights

    def initial_hidden_state(self):
        # batch_size is passed by the input shape in case the last batch is not exact the samp as self.batch_size
        return tf.zeros((self.batch_size, self.hidden_dim))





