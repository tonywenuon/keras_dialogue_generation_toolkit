import keras
from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects
from keras.layers import Input, Softmax, Embedding, Add, Lambda, Dense, RepeatVector, Layer

from models.keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from models.keras_transformer.masks import PaddingMaskLayer, SequenceMaskLayer
from models.keras_transformer.position import TransformerCoordinateEmbedding
from models.keras_transformer.transformer_blocks import TransformerACT, TransformerEncoderBlock, TransformerDecoderBlock

class UniversalTransformerModel:
    def __init__(self, args, 
                 transformer_dropout: float = 0.05,
                 embedding_dropout: float = 0.05,
                 l2_reg_penalty: float = 1e-4,
                 use_same_embedding = True,
                 use_vanilla_transformer = False,
                 ):
        self.args = args
        self.transformer_dropout = transformer_dropout 
        self.embedding_dropout = embedding_dropout

        # prepare layers
        l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
        if use_same_embedding:
            self.encoder_embedding_layer = self.decoder_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='embeddings',
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                # https://arxiv.org/pdf/1508.03721.pdf
                embeddings_regularizer=l2_regularizer)
        else:
            self.encoder_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='encoder_embeddings',
                embeddings_regularizer=l2_regularizer)
            self.decoder_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='decoder_embeddings',
                embeddings_regularizer=l2_regularizer)

        self.output_layer = TiedOutputEmbedding(
            projection_dropout=self.embedding_dropout,
            scaled_attention=True,
            projection_regularizer=l2_regularizer,
            name='word_prediction_logits')
        self.output_softmax_layer = Softmax(name='word_predictions')

        self.encoder_coord_embedding_layer = TransformerCoordinateEmbedding(
            self.args.src_seq_length,
            1 if use_vanilla_transformer else self.args.transformer_depth,
            name='encoder_coordinate_embedding')
        self.decoder_coord_embedding_layer = TransformerCoordinateEmbedding(
            self.args.tar_seq_length,
            1 if use_vanilla_transformer else self.args.transformer_depth,
            name='decoder_coordinate_embedding')

    def __get_encoder(self, input_layer):
        print('This is in Encoder...')
        self_attn_mask = PaddingMaskLayer(name='encoder_self_padding_mask', src_len=self.args.src_seq_length,
                                          pad_id=self.pad_id)(input_layer)

        next_step_input, _ = self.encoder_embedding_layer(input_layer)
        act_layer = TransformerACT(name='adaptive_computation_time_encoder')
        encoder_block = TransformerEncoderBlock(
                    name='transformer_encoder', 
                    num_heads=self.args.num_heads,
                    residual_dropout=self.transformer_dropout,
                    attention_dropout=self.transformer_dropout,
                    activation='relu',
                    vanilla_wiring=False) 

        for i in range(self.args.transformer_depth):
            next_step_input = self.encoder_coord_embedding_layer(next_step_input, step=i)
            next_step_input = encoder_block([next_step_input, self_attn_mask])
            next_step_input, act_output = act_layer(next_step_input)
        act_layer.finalize()
        next_step_input = act_output
        return next_step_input

    def __get_decoder(self, input_layer, encoder_output, mutual_attn_mask):
        print('This is in Decoder...')
        self_padding_mask = PaddingMaskLayer(name='decoder_self_padding_mask', src_len=self.args.tar_seq_length,
                                             pad_id=self.pad_id)(input_layer)
        seq_mask = SequenceMaskLayer(name='decoder_sequence_mask')(input_layer)
        self_attn_mask = Add()([self_padding_mask, seq_mask])
        # greater than 1, means not be padded in both self_padding_mask and seq_mask
        self_attn_mask = Lambda(lambda x: K.cast(K.greater((x), 1), dtype='int32'), name='decoder_add_padding_seq_mask')(self_attn_mask)

        next_step_input, self.decoder_embedding_matrix = self.decoder_embedding_layer(input_layer)
        act_layer = TransformerACT(name='adaptive_computation_time_decoder')
        decoder_block = TransformerDecoderBlock(
                    name='transformer_decoder', 
                    num_heads=self.args.num_heads,
                    residual_dropout=self.transformer_dropout,
                    attention_dropout=self.transformer_dropout,
                    activation='relu',
                    vanilla_wiring=False) 

        for i in range(self.args.transformer_depth):
            next_step_input = self.decoder_coord_embedding_layer(next_step_input, step=i)
            next_step_input = decoder_block([encoder_output, next_step_input, self_attn_mask, mutual_attn_mask])
            next_step_input, act_output = act_layer(next_step_input)
        act_layer.finalize()
        next_step_input = act_output

        return next_step_input

    def get_model(self, pad_id):
        self.pad_id = pad_id
        inp_src = Input(name='src_input',
                      shape=(None, ), 
                      dtype='int32'
                     )
        inp_answer = Input(name='answer_input',
                            shape=(None, ), 
                            dtype='int32',
                           )

        encoder_output = self.__get_encoder(inp_src)

        mutual_attn_mask = PaddingMaskLayer(name='decoder_mutual_padding_mask', src_len=self.args.tar_seq_length,
                                            pad_id=self.pad_id)(inp_src)

        decoder_output = self.__get_decoder(inp_answer, encoder_output, mutual_attn_mask)

        # build model part
        word_predictions = self.output_softmax_layer(
                self.output_layer([decoder_output, self.decoder_embedding_matrix]))
        model = Model(
                      inputs=[inp_src, inp_answer],
                      outputs=[word_predictions]
                     )
        return model

get_custom_objects().update({
    'PaddingMaskLayer': PaddingMaskLayer,
    'SequenceMaskLayer': SequenceMaskLayer,
    'UniversalTransformerModel': UniversalTransformerModel,
})
