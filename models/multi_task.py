"""Keras End-To-End Memory Networks.
The implementation is based on http://arxiv.org/abs/1503.08895
"""

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.layers import Lambda, Embedding, Layer, TimeDistributed, GRU, CuDNNGRU
from keras.layers import Dot, Activation, Dense, Reshape, Add
from keras.models import Model
from keras.utils import get_custom_objects
from models.keras_transformer.obtain_layer import GetLayer

def my_gru(name, units):
    return GRU(name=name,
                       units=units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform',
                       recurrent_activation='sigmoid')

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
        elif self._type == 'fact':
            pe_encoding = np.expand_dims(self.pe_encoding, 0)
            res = inputs * pe_encoding
        return res

class MemNNEncoder:
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.query_pe_embedding = PosEncodeEmbedding('query', 
            args.src_seq_length, 
            args.embedding_dim, 
            name='%s_query_pe'%self.name)
        self.fact_pe_embedding = PosEncodeEmbedding('fact', 
            args.src_seq_length, 
            args.embedding_dim, 
            name='%s_fact_pe'%self.name)

    #we have to turn sentence into embedding not just a word
    #here we just adding embeddings up
    def __emb_sent_bow(self, inp):
        emb_op = TimeDistributed(Embedding(self.args.vocab_size, self.args.embedding_dim, name='%s_embeddings'%self.name))
        emb = emb_op(inp) #taking each word and putting it through separate embedding, this is what Timedistributed doing 
        emb = self.fact_pe_embedding(emb)  # add position encoding to fact
        emb = Lambda(lambda x: K.sum(x, 2))(emb)  #then we do lambda layer to add them up
        return emb, emb_op

    def __one_hop(self, emb_q, A, inp_fact):
        # calculate weights between query and stories
        x = Reshape((1, self.args.embedding_dim))(emb_q)
        x = Dot(axes=2)([A, x])
        x = Reshape((self.args.fact_number, ))(x)
        x = Activation('softmax')(x)
        match = Reshape((self.args.fact_number, 1))(x)

        # multiply weights to stories
        emb_fact, _ = self.__emb_sent_bow(inp_fact)
        x = Dot(axes=1)([match, emb_fact])
        x = Reshape((self.args.embedding_dim, ))(x)
        x = Dense(self.args.embedding_dim)(x)
        # update query_embedding
        new_q = Add()([x, emb_q])
        return new_q, emb_fact

    def compute_output_shape(self, input_shape):
        q_shape = input_shape[0]
        return (q_shape[0], q_shape[1], self.args.embedding_dim)

    def __call__(self, inputs):
        inp_q, inp_fact= inputs[0], inputs[1]
        # query and stories share the first layer parameter
        emb_fact, emb_fact_op = self.__emb_sent_bow(inp_fact)
        emb_q = emb_fact_op.layer(inp_q)
        emb_q = self.query_pe_embedding(emb_q)
        emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)  #then we do lambda layer to add them up

        # from 2nd layer on, update query_embedding. Each hop, fact has new embeddings
        for i in range(self.args.hops):
            if i == 0:
                # the first layer
                response, emb_fact = self.__one_hop(emb_q, emb_fact, inp_fact)
            else:
                response, emb_fact = self.__one_hop(response, emb_fact, inp_fact)

        return response

class S2SEncoder:
    def __init__(self, args, **kwargs):
        self.args = args
        self.embedding = Embedding(args.vocab_size, args.embedding_dim, name='s2s_embeddings')
        self.gru = my_gru(name='s2s_encoder_gru', units=self.args.hidden_dim)
        self.hidden_dim = self.args.hidden_dim

    def __call__(self, inputs):
        inp_q = inputs
        emb_q = self.embedding(inp_q)
        output, state = self.gru(emb_q)
        return [output, state]

class MultiTaskModel:
    def __init__(self, args):
        self.args = args
        self.s2s_encoder = S2SEncoder(args, name='s2s_encoder')
        self.memnn_encoder1 = MemNNEncoder(args, name='memnn_encoder1')
        self.memnn_encoder2 = MemNNEncoder(args, name='memnn_encoder2')

        self.s2s_state_obtain_layer = GetLayer(name='s2s_state_obtain_layer')
        self.s2s_output_obtain_layer = GetLayer(name='s2s_output_obtain_layer')

        self.decoder = my_gru(name='decoder_gru', units=args.hidden_dim)
        self.decoder_dense1 = Dense(args.vocab_size, activation='softmax', name='od1')
        self.decoder_dense2 = Dense(args.vocab_size, activation='softmax', name='od2')
        self.decoder_dense3 = Dense(args.vocab_size, activation='softmax', name='od3')

        self.decoder_embedding = Embedding(name='decoder_embedding',
                                           input_dim=self.args.vocab_size, 
                                           output_dim=self.args.embedding_dim)

    def get_model(self):

        inp_fact = Input(name='fact_input', 
                            shape=(self.args.fact_number, self.args.src_seq_length), 
                            dtype='int32'
                           )
        inp_q = Input(name='query_input',
                            shape=(self.args.src_seq_length, ), 
                            dtype='int32'
                           )
        
        inp_tar = Input(name='tar_input',
                            shape=(self.args.tar_seq_length, ), 
                            dtype='int32',
                           )
        inp_fact_tar = Input(name='fact_tar_input',
                            shape=(self.args.tar_seq_length, ), 
                            dtype='int32',
                           )

        enc_output1, enc_state1 = self.s2s_encoder(inp_q)
        enc_state1 = self.s2s_state_obtain_layer(enc_state1)
        enc_output1 = self.s2s_output_obtain_layer(enc_output1)
        enc_state2 = self.memnn_encoder1([inp_q, inp_fact])
        enc_state3 = self.memnn_encoder2([inp_q, inp_fact])

        emb_ans = self.decoder_embedding(inp_tar)
        emb_fact_ans = self.decoder_embedding(inp_fact_tar)

        # task 1: seq2seq, input: question; output: answer
        output1, state1 = self.decoder(emb_ans, initial_state=enc_state1)
        # task 2: memnn, input: question and facts; output: fact
        output2, state2 = self.decoder(emb_fact_ans, initial_state=enc_state2)
        # task 3: memnn, input: question and facts; output: answer
        output3, state3 = self.decoder(emb_ans, initial_state=enc_state3)

        # final output
        final_output1 = self.decoder_dense1(output1)
        final_output2 = self.decoder_dense2(output2)
        final_output3 = self.decoder_dense3(output3)

        # define model
        model = Model(
            inputs=[inp_q, inp_tar, inp_fact_tar, inp_fact], 
            outputs=[final_output1, final_output2, final_output3]
        )

        return model

get_custom_objects().update({
    'PosEncodeEmbedding': PosEncodeEmbedding,
    'MultiTaskModel': MultiTaskModel,
    'S2SEncoder': S2SEncoder,
    'MemNNEncoder': MemNNEncoder,
})


