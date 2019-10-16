import numpy as np
# noinspection PyPep8Naming
from keras import backend as K
#from keras.engine import Layer
from keras.layers import Layer
from keras.utils import get_custom_objects


class GetLayer(Layer):
    """
    """
    def __init__(self, 
                 **kwargs):
        """
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs

get_custom_objects().update({
    'GetLayer': GetLayer,
})

