
import keras
from keras.layers import Lambda, Add
from keras.layers import Layer
from keras import backend as K
from keras.utils import get_custom_objects


class ProbCalcLayer(Layer):
    """get probability of generate from source layer.
    """
            
    def __init__(self, 
                 src_facts_number,
                 dropout: float = 0.0,
                 activation='relu', 
                 kernel_initializer='glorot_normal', 
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None,
                 **kwargs): 
        """Initialize the layer.
        :param src_facts_number: total number of src and facts.
        :param activation: Activations for linear mappings.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        """
        self.src_facts_number = src_facts_number
        self.prob_numbers = self.src_facts_number + 1
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        super(ProbCalcLayer, self).__init__(**kwargs)

    def get_config(self):
        config = { 
            'src_facts_number': self.src_facts_number,
            'dropout': self.dropout,
            'activation': keras.activations.serialize(self.activation), 
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer), 
            'bias_initializer': keras.initializers.serialize(self.bias_initializer), 
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer), 
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer), 
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint), 
            'bias_constraint': keras.constraints.serialize(self.bias_constraint), 
        } 
        base_config = super(ProbCalcLayer, self).get_config() 
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            self_shape = input_shape[1]
            return self_shape
        return input_shape

    def build(self, input_shape):
        if isinstance(input_shape, list):
            # sf: source and facts, k: self-attention
            sf, s = input_shape
        else:
            raise ValueError('Invalid input value. Expect a list, but %s is received' % (type(input_shape)))

        hidden_dim = int(s[-1])

        # source, facts and self_attention Tensor
        self.Wsf = self.add_weight(
            shape=(self.src_facts_number, hidden_dim, self.prob_numbers),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wsf' % (self.name),
        )
        self.Ws = self.add_weight(
            shape=(hidden_dim, self.prob_numbers),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Ws' % self.name,
        )
        self.b = self.add_weight(
            shape=(self.prob_numbers,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            name='%s_b' % self.name,
        )
        super(ProbCalcLayer, self).build(input_shape)

    def apply_dropout_if_needed(self, _input, training=None):
        if 0.0 < self.dropout < 1.0:
            def dropped_softmax():
                return K.dropout(_input, self.dropout)

            return K.in_train_phase(dropped_softmax, _input,
                                    training=training)
        return _input

    def call(self, inputs):
        if isinstance(inputs, list) and len(inputs) == 2:
            # shape (bs, sf_number, len, dim)
            sf_outputs, self_output = inputs
        else:
            raise ValueError('Expect list for ProbCalcLayer, but receiced %s' % type(inputs))

        print('in ProbCalcLayer......')
        self_shape = K.shape(self_output)
        sf_shape = K.shape(sf_outputs)
        hidden_dim = self_shape[-1]

        self_calc = K.permute_dimensions(self_output, [1, 0, 2, 3])
        self_dot = K.dot(K.reshape(self_calc, (-1, hidden_dim)), self.Ws) # shape: (1, bs, seq_len, src_fact_number + 1)
        probs = self_dot

        # shape: (sf_number, bs, seq_len, hidden_dim)
        sf_calc = K.permute_dimensions(sf_outputs, [1, 0, 2, 3])

        # shape: (sf_number, bs, seq_len, hidden_dim)
        for i in range(self.src_facts_number):
            sf_dot = K.dot(K.reshape(sf_calc[i], (-1, hidden_dim)), self.Wsf[i])
            probs += sf_dot

        probs = K.bias_add(probs, self.b)
        probs = K.sigmoid(probs)
        probs = self.apply_dropout_if_needed(probs)
        probs = K.reshape(probs, (self_shape[0], self_shape[2], self.prob_numbers, 1))
        probs = K.softmax(probs)

        # merge together
        # shape: (bs, sf_number + 1, seq_len, hidden_dim)
        src_facts_self_tensor = K.concatenate([sf_outputs, self_output], axis=1) 
        # shape: (bs, seq_len, sf_number + 1, hidden_dim)
        src_facts_self_tensor = K.permute_dimensions(src_facts_self_tensor, [0, 2, 1, 3])
        src_facts_self_tensor = K.cast(src_facts_self_tensor, dtype=K.floatx())
        src_facts_self_tensor = src_facts_self_tensor * probs
        result = K.sum(src_facts_self_tensor, axis=-2) # shape: (bs, seq_len, hidden_dim)
        result = K.expand_dims(result, axis=1)

        return result

get_custom_objects().update({
    'ProbCalcLayer': ProbCalcLayer,
})

