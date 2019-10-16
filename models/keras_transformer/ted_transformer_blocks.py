"""
Contains implementation of the Transformer model described in papers
"Attention is all you need" (https://arxiv.org/abs/1706.03762) and
"Universal Transformer" (https://arxiv.org/abs/1807.03819)
"""
import math
from typing import Union, Callable, Optional

from keras.layers import Layer, Add, Dropout, Lambda, Reshape, Concatenate
from keras import initializers, activations
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects

from models.keras_transformer.ted_attention import TEDMultiHeadSelfAttention
from models.keras_transformer.ted_attention import TEDMultiHeadAttention
from models.keras_transformer.normalization import LayerNormalization
from models.keras_transformer.ted_transition import TEDTransformerTransition
from models.keras_transformer.ted_src_facts_merge import ProbCalcLayer

class TEDEncoderBlock:
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each such block is, essentially:

    - Multi-head self-attention (masked or unmasked, with attention dropout,
      but without input dropout)
    - Residual connection,
    - Dropout
    - Layer normalization
    - Transition function
    - Residual connection
    - Dropout
    - Layer normalization

    Also check TransformerACT class if you need support for ACT (Adaptive
    Computation Time).

    IMPORTANT: The older Transformer 2017 model ("Attention is all you need")
    uses slightly different order of operations. A quote from the paper:

        "We apply dropout [33] to the output of each sub-layer,
         before it is added to the sub-layer input and normalized"

    while the Universal Transformer paper puts dropout one step *after*
    the sub-layers's output was added to its input (Figure 4 in the paper).

    In this code the order from the Universal Transformer is used, as arguably
    more reasonable. You can use classical Transformer's (2017) way of
    connecting the pieces by passing vanilla_wiring=True to the constructor.
    """
    def __init__(self, name: str, num_heads: int, 
                 elem_number: int,
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 vanilla_wiring=True):
        self.self_attention_layer = TEDMultiHeadSelfAttention(
            num_heads, 
            elem_number,
            dropout=attention_dropout,
            name=f'{name}_self_attention')
        self.norm1_layer = LayerNormalization(name=f'{name}_norm1')
        self.dropout_layer = (
            Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.norm2_layer = LayerNormalization(name=f'{name}_norm2')
        self.transition_layer = TEDTransformerTransition(name=f'{name}_transition', activation=activation)
        self.add_layer = Add(name=f'{name}_add')
        self.vanilla_wiring = vanilla_wiring

    def __call__(self, inputs):
        _input, self_attn_mask = inputs
        output = self.self_attention_layer([_input, self_attn_mask])

        # by default, self.vanilla_writing is False, so go else branch
        post_residual1 = (
            self.add_layer([_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.add_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)

        output = self.transition_layer(norm1_output)
        post_residual2 = (
            self.add_layer([norm1_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.add_layer([norm1_output, output])))
        output = self.norm2_layer(post_residual2)
        print('norm2: ', output)
        return output

class TEDDecoderBlock:
    """
    A pseudo-layer combining together all nuts and bolts to assemble
    a complete section of both the Transformer and the Universal Transformer
    models, following description from the "Universal Transformers" paper.
    Each such block is, essentially:

    - Multi-head self-attention (masked or unmasked, with attention dropout,
      but without input dropout)
    - Residual connection,
    - Dropout
    - Layer normalization
    - Multi-head attention (between decoder input and encoder output)
    - Residual connection,
    - Dropout
    - Layer normalization
    - Transition function
    - Residual connection
    - Dropout
    - Layer normalization

    Also check TransformerACT class if you need support for ACT (Adaptive
    Computation Time).

    IMPORTANT: The older Transformer 2017 model ("Attention is all you need")
    uses slightly different order of operations. A quote from the paper:

        "We apply dropout [33] to the output of each sub-layer,
         before it is added to the sub-layer input and normalized"

    while the Universal Transformer paper puts dropout one step *after*
    the sub-layers's output was added to its input (Figure 4 in the paper).

    In this code the order from the Universal Transformer is used, as arguably
    more reasonable. You can use classical Transformer's (2017) way of
    connecting the pieces by passing vanilla_wiring=True to the constructor.
    """
    def __init__(self, name: str, num_heads: int, 
                 fact_number: int,
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 vanilla_wiring=True):
        print('This is in %s...' % name)
        self.self_attention_layer = TEDMultiHeadSelfAttention(
            num_heads, 
            1,
            dropout=attention_dropout,
            name=f'{name}_self_attention')
        self.mutual_attention_layer = TEDMultiHeadAttention(
            num_heads, 
            fact_number + 1,
            dropout=attention_dropout,
            name=f'{name}_mutual_attention')

        self.prob_calc_layer = ProbCalcLayer(
            fact_number + 1,
            dropout=attention_dropout,
            name=f'{name}_prob_calc')

        self.fact_number = fact_number
        self.norm1_layer = LayerNormalization(name=f'{name}_norm1')
        self.norm2_layer = LayerNormalization(name=f'{name}_norm2')
        self.norm3_layer = LayerNormalization(name=f'{name}_norm3')

        self.dropout_layer = (
            Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.transition_layer = TEDTransformerTransition(name=f'{name}_transition', activation=activation)
        self.add_layer = Add(name=f'{name}_add')
        self.vanilla_wiring = vanilla_wiring

    def __call__(self, inputs):
        decoder_input, self_attn_mask, \
        src_encoder_output, mutual_tar_src_mask, \
        fact_encoder_output, mutual_tar_fact_mask = inputs

        #print('[[[start]]]')
        #print('decoder_input: ', decoder_input)
        #print('src_encoder_output: ', src_encoder_output)
        #print('mutual_tar_src_mask: ', mutual_tar_src_mask)
        #print('fact_encoder_output: ', fact_encoder_output)
        #print('mutual_tar_fact_mask: ', mutual_tar_fact_mask)
        #print('[[[end start]]]')

        # fact_encoder_output: (bs, f_num, f_len, dim)
        f_shape = K.shape(fact_encoder_output)
        # mutual_tar_fact_mask: (bs, f_num, tar_len, f_len)
        f_mask_shape = K.shape(mutual_tar_fact_mask)

        # self attention part
        output = self.self_attention_layer([decoder_input, self_attn_mask])
        post_residual1 = (
            self.add_layer([decoder_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.add_layer([decoder_input, output])))
        norm1_output = self.norm1_layer(post_residual1)

        # mutual attention part
        src_fact_encoder_output = Concatenate(axis=1)([src_encoder_output, fact_encoder_output])
        mutual_tar_src_fact_mask = Concatenate(axis=1)([mutual_tar_src_mask, mutual_tar_fact_mask])

        # shape: (bs, sf_number, tar_len, dim)
        output = self.mutual_attention_layer([src_fact_encoder_output, norm1_output, mutual_tar_src_fact_mask])
        output = self.prob_calc_layer([output, norm1_output])

        post_residual2 = (
            self.add_layer([norm1_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.add_layer([norm1_output, output])))
        norm2_output = self.norm2_layer(post_residual2)

        output = self.transition_layer(norm2_output)
        post_residual3 = (
            self.add_layer([norm2_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.add_layer([norm2_output, output])))
        output = self.norm3_layer(post_residual3)

        return output

get_custom_objects().update({
    'TEDEncoderBlock': TEDEncoderBlock,
    'TEDDecoderBlock': TEDDecoderBlock,
})
