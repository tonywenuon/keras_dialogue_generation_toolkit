import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
#from keras.engine import Layer
from keras.layers import Layer
from keras.utils import get_custom_objects


class _BaseTEDMultiHeadAttention(Layer):
    """
    Base class for two types of Multi-head attention layers:
    Self-attention and its more general form used in decoders (the one which
    takes values and keys from the encoder).
    """
    def __init__(self, num_heads: int, 
                 src_facts_number: int,
                 use_masking: bool = False,
                 dropout: float = 0.0,
                 **kwargs):
        """
        :param num_heads: number of attention heads
        use_masking para is not used any more.
        :param use_masking: when True, forbids the attention to see the further
          elements in the sequence (particularly important in language
          modelling).
        :param dropout: dropout that should be applied to the attention
          (after the softmax).
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.num_heads = num_heads
        self.src_facts_number = src_facts_number
        self.use_masking = use_masking
        self.dropout = dropout
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['src_facts_number'] = self.src_facts_number
        config['use_masking'] = self.use_masking
        config['dropout'] = self.dropout
        return config

    # noinspection PyAttributeOutsideInit
    def build_output_params(self, d_model):
        self.output_weights = self.add_weight(
            name='output_weights',
            shape=(d_model, d_model),
            initializer='glorot_normal',
            trainable=True)

    def validate_model_dimensionality(self, d_model: int):
        if d_model % self.num_heads != 0:
            raise ValueError(
                f'The size of the last dimension of the input '
                f'({d_model}) must be evenly divisible by the number'
                f'of the attention heads {self.num_heads}')

    def attention(self, pre_q, pre_v, pre_k, out_seq_len, d_model, attn_mask,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, src_facts_number, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, src_facts_number, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, src_facts_number, k_seq_len, num_heads, d_model // num_heads)
        :param out_seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        # pre q, v, k (batch_size, sf_number, seq_len, num_heads, d_model//heads)

        #print('in attention function.......................')
        #print('pre_q: ', pre_q)
        #print('pre_k: ', pre_k)
        #print('pre_v: ', pre_v)
        # shaping Q and V into (batch_size*sf_number, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])
        k_transposed = K.permute_dimensions(pre_k, [0, 2, 3, 1])

        # for further matrix multiplication
        sqrt_d = K.sqrt(K.cast(d_model, dtype=K.floatx()) // self.num_heads)
        q_shape = K.shape(q)
        v_shape = K.shape(v)
        k_t_shape = K.shape(k_transposed)

        # before performing batch_dot all tensors are being converted to 3D
        # performs identically on all backends
        reshape1 = K.reshape(q, (-1, q_shape[-2], q_shape[-1]))
        reshape2 = K.reshape(k_transposed, (-1, k_t_shape[-2], k_t_shape[-1]))
        # mask the attention for the prediction process
        mask_attention = self.mask_attention(
                            # core scaled dot product
                            K.batch_dot( 
                                reshape1, 
                                reshape2)
                            / sqrt_d, attn_mask)
        attention_heads = K.reshape(
            K.batch_dot(
                self.apply_dropout_if_needed(
                    mask_attention,
                    training=training),
                K.reshape(v, (-1, v_shape[-2], v_shape[-1]))),
            (-1, self.num_heads, q_shape[-2], q_shape[-1]))
        attention_heads_merged = K.reshape(
            K.permute_dimensions(attention_heads, [0, 2, 1, 3]),
            (-1, d_model))
        attention_out = K.reshape(
            K.dot(attention_heads_merged, self.output_weights),
            (-1, self.src_facts_number, out_seq_len, d_model))
        return attention_out

    def apply_dropout_if_needed(self, attention_softmax, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(attention_softmax, self.dropout)

            return K.in_train_phase(dropped_softmax, attention_softmax,
                                    training=training)
        return attention_softmax

    def mask_attention(self, dot_product, attn_mask, training=None):
        #print('in %s...' % self.name)
        #print('dot_product1: ', dot_product)
        #print('attn_mask1: ', attn_mask)
        dot_shape = K.shape(dot_product)
        q_len, k_len = dot_shape[-2], dot_shape[-1]

        dot_product = K.reshape(dot_product, (-1, self.num_heads, q_len, k_len))
        dot_product = K.permute_dimensions(dot_product, [1, 0, 2, 3])

        attn_mask = K.reshape(attn_mask, (-1, q_len, k_len))
        attn_mask = K.cast(attn_mask, dtype=K.floatx()) 
        dot_product = dot_product * attn_mask

        close_to_negative_inf = K.constant(-1e9, dtype=K.floatx())
        negative_inf_matrix = 1 - attn_mask
        negative_inf_matrix *= close_to_negative_inf
        dot_product += negative_inf_matrix

        dot_product = K.softmax(dot_product)
        dot_product = dot_product * attn_mask
        dot_product = K.permute_dimensions(dot_product, [1, 0, 2, 3])
        dot_product = K.reshape(dot_product, (-1, q_len, k_len))
        return dot_product
            

class TEDMultiHeadAttention(_BaseTEDMultiHeadAttention):
    """
    Multi-head attention which can use two inputs:
    First: from the encoder - it's used to project the keys and the values
    Second: from the decoder - used to project the queries.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not (isinstance(input_shape, list) and len(input_shape) == 3):
        #if not (isinstance(input_shape, list) and len(input_shape) == 2):
            raise ValueError(
                'You must call this layer passing a list of three tensors'
                '(for keys/values and queries)')
        value_dim, query_dim = input_shape[0][-1], input_shape[1][-1]
        if query_dim != value_dim:
            raise ValueError(
                f'Both keys/value and query inputs must be '
                f'of the same dimensionality, instead of '
                f'{value_dim} and {query_dim}.')
        d_model = query_dim
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_k and W_v which
        # are, in turn, concatenated W matrices of keys, and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.

        self.kv_weights = self.add_weight(
            name='kv_weights', shape=(d_model, d_model * 2),
            initializer='glorot_normal', trainable=True)
        self.q_weights = self.add_weight(
            name='q_weights', shape=(d_model, d_model),
            initializer='glorot_normal', trainable=True)
        self.build_output_params(d_model)

        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def call(self, inputs, **kwargs):
        if not (isinstance(inputs, list) and len(inputs) == 3):
        #if not (isinstance(inputs, list) and len(inputs) == 2):
            raise ValueError(
                'You can call this layer only with a list of three tensors '
                '(for keys/values and queries)')
        key_value_input, query_input, mutual_attn_mask = inputs
        key_values_shapes = K.shape(key_value_input)
        value_seq_len, d_model = key_values_shapes[-2], key_values_shapes[-1]

        query_seq_len = K.shape(query_input)[-2]
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        kv = K.dot(K.reshape(key_value_input, [-1, d_model]), self.kv_weights)
        # splitting the keys, the values and the queries before further
        # processing
        pre_k, pre_v = [
            K.reshape(
                # K.slice(kv, (0, i * d_model), (-1, d_model)),
                kv[:, i * d_model: (i + 1) * d_model],
                (-1, value_seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(2)]
        pre_q = K.reshape(
            K.dot(K.reshape(query_input, [-1, d_model]), self.q_weights),
            (-1, 1, query_seq_len, self.num_heads, d_model // self.num_heads))
        pre_q = K.repeat_elements(pre_q, self.src_facts_number, axis = 1) 
        pre_q = K.reshape(pre_q, (-1, query_seq_len, self.num_heads, d_model // self.num_heads))
        return self.attention(pre_q, pre_v, pre_k, query_seq_len, d_model, mutual_attn_mask,
                              training=kwargs.get('training'))


class TEDMultiHeadSelfAttention(_BaseTEDMultiHeadAttention):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input and has implementation which is better suited for
    such use case that more general MultiHeadAttention class.
    """

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not isinstance(input_shape, list):
        #if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')
        d_model = input_shape[0][-1]
        #d_model = input_shape[-1]

        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_q, W_k and W_v which
        # are, in turn, concatenated W matrices of keys, queries and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_normal',
            trainable=True)
        self.build_output_params(d_model)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not (K.is_tensor(inputs[0]) and K.is_tensor(inputs[1])):
        #if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        query_input, self_attn_mask = inputs

        query_shape = K.shape(query_input)
        seq_len, d_model = query_shape[-2], query_shape[-1]
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        qkv = K.dot(K.reshape(query_input, [-1, d_model]), self.qkv_weights) # shape: (batch_size, seq_len, 3 * d_model)
        # splitting the keys, the values and the queries before further
        # processing
        pre_q, pre_k, pre_v = [
            K.reshape(
                qkv[:, i * d_model:(i + 1) * d_model],
                # if there is error, the shape is wrong
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        attention_out = self.attention(pre_q, pre_v, pre_k, seq_len, d_model, self_attn_mask,
                                       training=kwargs.get('training'))
        return attention_out

    def compute_output_shape(self, input_shape):
        return input_shape[0]


get_custom_objects().update({
    'TEDMultiHeadSelfAttention': TEDMultiHeadSelfAttention,
    'TEDMultiHeadAttention': TEDMultiHeadAttention,
})

