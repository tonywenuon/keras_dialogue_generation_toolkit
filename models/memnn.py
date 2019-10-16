"""Keras End-To-End Memory Networks.
The implementation is based on http://arxiv.org/abs/1503.08895
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda, Embedding, Layer, TimeDistributed, GRU, CuDNNGRU
from tensorflow.keras.layers import Dot, Activation, Dense, Reshape, Add
from tensorflow.keras.models import Model

__all__ = [
    'my_gru',
    'get_custom_objects',
]

def get_custom_objects():
    return {
        'PosEncodeEmbedding': PosEncodeEmbedding,
        'MemNNModel': MemNNModel,
    }

class PosEncodeEmbedding(Layer):
    """
    Position Encoding described in section 4.1
    """
    def __init__(self, _type, seq_len, embedding_dim, **kwargs):
        self._type = _type
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.pe_encoding = self.__position_encoding()
        super(PosEncodeEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['_type'] = self._type
        config['seq_len'] = self.seq_len
        config['embedding_dim'] = self.embedding_dim
        config['pe_encoding'] = self.pe_encoding
        return config 

    def compute_output_shape(self, input_shape):
        return input_shape

    def __position_encoding(self):
        encoding = np.ones((self.embedding_dim, self.seq_len), dtype=np.float32)
        ls = self.seq_len + 1
        le = self.embedding_dim + 1
        for i in range(1, le):
            for j in range(1, ls):
                encoding[i-1, j-1] = (i - (self.embedding_dim + 1) / 2) * (j - (self.seq_len + 1) / 2)
        encoding = 1 + 4 * encoding / self.embedding_dim / self.seq_len
        # Make position encoding of time words identity to avoid modifying them 
        encoding[:, -1] = 1.0
        # shape: (seq_len, embedding_dim)
        return np.transpose(encoding)

    def call(self, inputs):
        assert len(inputs.shape) >= 2
        res = None
        if self._type == 'query':
            res = inputs * self.pe_encoding
        elif self._type == 'story':
            pe_encoding = np.expand_dims(self.pe_encoding, 0)
            res = inputs * pe_encoding
        return res

def my_gru(name, units):
        # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
        # the code automatically does that.
        if tf.test.is_gpu_available():
            print('### This is CuDNNGRU ###')
            return CuDNNGRU(name=name,
                            units=units,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform')
        else:
            return GRU(name=name,
                       units=units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform',
                       recurrent_activation='sigmoid')

class MemNNModel:
    def __init__(self, args):
        self.args = args
        self.query_pe_embedding = PosEncodeEmbedding('query', args.src_seq_length, args.embedding_dim)
        self.story_pe_embedding = PosEncodeEmbedding('story', args.fact_seq_length, args.embedding_dim)
        self.decoder = my_gru(name='decoder_gru', units=args.hidden_dim)
        self.decoder_dense = Dense(args.vocab_size, activation='softmax', name='output_dense')
        self.decoder_embedding = Embedding(name='decoder_embedding',
                                           input_dim=self.args.vocab_size, 
                                           output_dim=self.args.embedding_dim)

    def get_config(self):
        config = super().get_config()
        config['args'] = self.args
        return config

    #we have to turn sentence into embedding not just a word
    #here we just adding embeddings up
    def __emb_sent_bow(self, inp):
        emb_op = TimeDistributed(Embedding(self.args.vocab_size, self.args.embedding_dim, name='embeddings'))
        emb = emb_op(inp) #taking each word and putting it through separate embedding, this is what Timedistributed doing 
        emb = self.story_pe_embedding(emb)  # add position encoding to story
        emb = Lambda(lambda x: K.sum(x, 2))(emb)  #then we do lambda layer to add them up
        return emb, emb_op

    def get_model(self):
        def _one_hop(emb_q, A):
            # calculate weights between query and stories
            x = Reshape((1, self.args.embedding_dim))(emb_q)
            x = Dot(axes=2)([A, x])
            x = Reshape((self.args.fact_number, ))(x)
            x = Activation('softmax')(x)
            match = Reshape((self.args.fact_number, 1))(x)

            # multiply weights to stories
            emb_story, _ = self.__emb_sent_bow(inp_story)
            x = Dot(axes=1)([match, emb_story])
            x = Reshape((self.args.embedding_dim, ))(x)
            x = Dense(self.args.embedding_dim)(x)
            # update query_embedding
            new_q = Add()([x, emb_q])
            return new_q, emb_story

        inp_story = Input(name='story_input', 
                            shape=(self.args.fact_number, self.args.fact_seq_length), 
                            dtype='int32'
                           )
        inp_q = Input(name='query_input',
                            shape=(self.args.src_seq_length, ), 
                            dtype='int32'
                           )
        
        # query and stories share the first layer parameter
        emb_story, emb_story_op = self.__emb_sent_bow(inp_story)
        emb_q = emb_story_op.layer(inp_q)
        emb_q = self.query_pe_embedding(emb_q) 
        emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)

        # from 2nd layer on, update query_embedding. Each hop, story has new embeddings
        for i in range(self.args.hops):
            if i == 0:
                # the first layer
                response, emb_story = _one_hop(emb_q, emb_story)
            else:
                response, emb_story = _one_hop(response, emb_story)

        # get the final output of the MemNN, taking as input of the following GRU Decoder
        encoder_states = response
        inp_answers = Input(name='answer_input',
                            shape=(self.args.tar_seq_length, ), 
                            dtype='int32',
                           )
        emb_answers = self.decoder_embedding(inp_answers)
        outputs, states = self.decoder(emb_answers, initial_state=encoder_states)

        # final output
        final_outputs = self.decoder_dense(outputs)

        # define model
        model = Model(inputs=[inp_q, inp_story, inp_answers], outputs=final_outputs)
        model.summary()

        # Define the inference model: Encoder
        encoder_model = Model(inputs=[inp_q, inp_story], outputs=encoder_states)

        # Define the inference model: Decoder
        inp_decoder_state = Input(shape=(self.args.embedding_dim, ))
        decoder_outputs, decoder_states = self.decoder(emb_answers, initial_state=inp_decoder_state)
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(inputs=[inp_answers, inp_decoder_state], outputs=decoder_outputs)

        return model, encoder_model, decoder_model
        

