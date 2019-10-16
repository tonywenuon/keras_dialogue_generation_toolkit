"""
Contains implementation of the Transformer model described in papers
"Attention is all you need" (https://arxiv.org/abs/1706.03762) and
"Universal Transformer" (https://arxiv.org/abs/1807.03819)
"""
import math
from typing import Union, Callable, Optional

from keras.layers import Layer, Add, Dropout
from keras import initializers, activations
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects

from models.keras_transformer.attention import MultiHeadSelfAttention
from models.keras_transformer.attention import MultiHeadAttention
from models.keras_transformer.normalization import LayerNormalization
from models.keras_transformer.transition import TransformerTransition

def gelu(x):
    """
    GELU activation, described in paper "Gaussian Error Linear Units (GELUs)"
    https://arxiv.org/pdf/1606.08415.pdf
    """
    c = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + K.tanh(c * (x + 0.044715 * K.pow(x, 3))))

class TransformerEncoderBlock:
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
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 compression_window_size: int = None,
                 vanilla_wiring=False):
        self.self_attention_layer = MultiHeadSelfAttention(
            num_heads, dropout=attention_dropout,
            compression_window_size=compression_window_size, 
            name=f'{name}_self_attention')
        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')
        self.dropout_layer = (
            Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')
        self.transition_layer = TransformerTransition(name=f'{name}_transition', activation=activation)
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
        return output

class TransformerDecoderBlock:
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
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 compression_window_size: int = None,
                 vanilla_wiring=False):
        self.self_attention_layer = MultiHeadSelfAttention(
            num_heads, dropout=attention_dropout,
            compression_window_size=compression_window_size, 
            name=f'{name}_self_attention')
        self.mutual_attention_layer = MultiHeadAttention(
            num_heads, dropout=attention_dropout,
            compression_window_size=compression_window_size, 
            name=f'{name}_mutual_attention')

        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')
        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')
        self.norm3_layer = LayerNormalization(name=f'{name}_normalization3')

        self.dropout_layer = (
            Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.transition_layer = TransformerTransition(name=f'{name}_transition', activation=activation)
        self.add_layer = Add(name=f'{name}_add')
        self.vanilla_wiring = vanilla_wiring

    def __call__(self, inputs):
        encoder_output, decoder_input, self_attn_mask, mutual_attn_mask = inputs

        # self attention part
        output = self.self_attention_layer([decoder_input, self_attn_mask])

        # by default, self.vanilla_writing is False, so go else branch
        post_residual1 = (
            self.add_layer([decoder_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.add_layer([decoder_input, output])))
        norm1_output = self.norm1_layer(post_residual1)

        # mutual attention part
        output = self.mutual_attention_layer([encoder_output, norm1_output, mutual_attn_mask])
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

class TransformerBlock:
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
                 residual_dropout: float = 0, attention_dropout: float = 0,
                 activation: Optional[Union[str, Callable]] = 'relu',
                 compression_window_size: int = None,
                 vanilla_wiring=False):
        self.attention_layer = MultiHeadSelfAttention(
            num_heads, dropout=attention_dropout,
            compression_window_size=compression_window_size,
            name=f'{name}_self_attention')
        self.norm1_layer = LayerNormalization(name=f'{name}_normalization1')
        self.dropout_layer = (
            Dropout(residual_dropout, name=f'{name}_dropout')
            if residual_dropout > 0
            else lambda x: x)
        self.norm2_layer = LayerNormalization(name=f'{name}_normalization2')
        self.transition_layer = TransformerTransition(name=f'{name}_transition', activation=activation)
        self.addition_layer = Add(name=f'{name}_add')
        self.vanilla_wiring = vanilla_wiring

    def __call__(self, _input):
        output = self.attention_layer(_input)
        # by default, self.vanilla_writing is False, so go else branch
        post_residual1 = (
            self.addition_layer([_input, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(self.addition_layer([_input, output])))
        norm1_output = self.norm1_layer(post_residual1)
        output = self.transition_layer(norm1_output)
        post_residual2 = (
            self.addition_layer([norm1_output, self.dropout_layer(output)])
            if self.vanilla_wiring
            else self.dropout_layer(
                self.addition_layer([norm1_output, output])))
        output = self.norm2_layer(post_residual2)
        return output

class TransformerACT(Layer):
    """
    Implements Adaptive Computation Time (ACT) for the Transformer model
    https://arxiv.org/abs/1603.08983

    How to use:

        transformer_depth = 8
        block = TransformerBlock('Transformer', num_heads=8)
        act_layer = TransformerACT()
        next_input = input  # (batch_size, sequence_length, input_size)
        for i in range(transformer_depth):
            next_input = block(next_input, step=i)
            next_input, act_weighted_output = act_layer(next_input)
        act_layer.finalize()  # adds loss
        result = act_weighted_output

    """
    def __init__(self, halt_epsilon=0.01, time_penalty=0.01, **kwargs):
        """
        :param halt_epsilon: a small constant that allows computation to halt
            after a single update (sigmoid never reaches exactly 1.0)
        :param time_penalty: parameter that weights the relative cost
            of computation versus error. The larger it is, the less
            computational steps the network will try to make and vice versa.
            The default value of 0.01 works well for Transformer.
        :param kwargs: Any standard parameters for a layer in Keras (like name)
        """
        self.halt_epsilon = halt_epsilon
        self.time_penalty = time_penalty
        self.ponder_cost = None
        self.weighted_output = None
        self.zeros_like_input = None
        self.zeros_like_halting = None
        self.ones_like_halting = None
        self.halt_budget = None
        self.remainder = None
        self.active_steps = None
        super().__init__(**kwargs)

    def get_config(self):
        return dict(
            super().get_config(),
            halt_epsilon=self.halt_epsilon,
            time_penalty=self.time_penalty)

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        assert len(input_shape) == 3
        # d_model: word_embedding_size
        _, sequence_length, d_model = input_shape
        self.halting_kernel = self.add_weight(
            name='halting_kernel',
            shape=(d_model, 1),
            initializer='glorot_uniform',
            trainable=True)
        self.halting_biases = self.add_weight(
            name='halting_biases',
            shape=(1,),
            initializer=initializers.Constant(0.1),
            trainable=True)
        self.time_penalty_t = K.constant(self.time_penalty, dtype=K.floatx())
        return super().build(input_shape)

    def initialize_control_tensors(self, halting):
        """
        Initializes constants and some step-tracking variables
        during the first call of the layer (since for the Universal Transformer
        all the following calls are supposed to be with inputs of identical
        shapes).
        """
        self.zeros_like_halting = K.zeros_like(
            halting, name='zeros_like_halting')
        self.ones_like_halting = K.ones_like(
            halting, name='ones_like_halting')
        self.remainder = self.ones_like_halting
        self.active_steps = self.zeros_like_halting
        self.halt_budget = self.ones_like_halting - self.halt_epsilon

    def call(self, inputs, **kwargs):
        #input_shape = K.int_shape(inputs)
        input_shape = K.shape(inputs)
        sequence_length, d_model = input_shape[-2], input_shape[-1]
        # output of the "sigmoid halting unit" (not the probability yet)
        halting = K.sigmoid(
            K.reshape(
                K.bias_add(
                    K.dot(K.reshape(inputs, [-1, d_model]),
                          self.halting_kernel),
                    self.halting_biases,
                    data_format='channels_last'),
                [-1, sequence_length]))
        if self.zeros_like_halting is None:
            self.initialize_control_tensors(halting)
        # useful flags
        step_is_active = K.greater(self.halt_budget, 0)
        no_further_steps = K.less_equal(self.halt_budget - halting, 0)
        # halting probability is equal to
        # a. halting output if this isn't the last step (we have some budget)
        # b. to remainder if it is,
        # c. and zero for the steps that shouldn't be executed at all
        #    (out of budget for them)
        halting_prob = K.switch(
            step_is_active,
            K.switch(
                no_further_steps,
                self.remainder,
                halting),
            self.zeros_like_halting)
        self.active_steps += K.switch(
            step_is_active,
            self.ones_like_halting,
            self.zeros_like_halting)
        # We don't know which step is the last, so we keep updating
        # expression for the loss with each call of the layer
        self.ponder_cost = (
            self.time_penalty_t * K.mean(self.remainder + self.active_steps))
        # Updating "the remaining probability" and the halt budget
        self.remainder = K.switch(
            no_further_steps,
            self.remainder,
            self.remainder - halting)
        self.halt_budget -= halting  # OK to become negative

        # If none of the inputs are active at this step, then instead
        # of zeroing them out by multiplying to all-zeroes halting_prob,
        # we can simply use a constant tensor of zeroes, which means that
        # we won't even calculate the output of those steps, saving
        # some real computational time.
        if self.zeros_like_input is None:
            self.zeros_like_input = K.zeros_like(
                inputs, name='zeros_like_input')
        # just because K.any(step_is_active) doesn't work in PlaidML
        any_step_is_active = K.greater(
            K.sum(K.cast(step_is_active, 'int32')), 0)
        step_weighted_output = K.switch(
            any_step_is_active,
            K.expand_dims(halting_prob, -1) * inputs,
            self.zeros_like_input)
        if self.weighted_output is None:
            self.weighted_output = step_weighted_output
        else:
            self.weighted_output += step_weighted_output
        return [inputs, self.weighted_output]

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

    def finalize(self):
        self.add_loss(self.ponder_cost)


get_custom_objects().update({
    'TransformerEncoderBlock': TransformerEncoderBlock,
    'TransformerDecoderBlock': TransformerDecoderBlock,
    'TransformerACT': TransformerACT,
    'gelu': gelu,
})
