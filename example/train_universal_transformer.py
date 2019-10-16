import os, sys, time, math

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import tensorflow as tf
import keras
import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import get_custom_objects
from models.universal_transformer import UniversalTransformerModel
from commonly_used_code import helper_fn, config
from run_script.args_parser import universal_transformer_add_arguments
from vspgt_data_reader import DataSet
import keras.backend.tensorflow_backend as KTF

#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class UniversalTransformer:
    def __init__(self, args):
        # real Transformer model architecture
        self.uni_transformer_model = UniversalTransformerModel(args=args,
                                                  transformer_dropout=0.05,
                                                  embedding_dropout=0.0,
                                                  use_same_embedding=False,
                                                  )
        self.args = args
        exp_name = args.data_set + '_' + args.exp_name

        # create experiment dir
        self.exp_dir= os.path.join(args.checkpoints_dir, exp_name)
        helper_fn.makedirs(self.exp_dir)
        hist_name = exp_name + '.hist'
        model_name = exp_name + '_final_model.h5'
        model_weights_name = exp_name + '_final_model_weights.h5'

        self.history_path = os.path.join(self.exp_dir, hist_name)
        self.model_path = os.path.join(self.exp_dir, model_name)
        self.model_weights_path = os.path.join(self.exp_dir, model_weights_name)
        
        outputs_dir = args.outputs_dir
        helper_fn.makedirs(outputs_dir)
        self.src_out_name = exp_name + '.src'
        self.src_out_path = os.path.join(outputs_dir, self.src_out_name)
        self.pred_out_name = exp_name + '.pred'
        self.pred_out_path = os.path.join(outputs_dir, self.pred_out_name)
        self.tar_out_name = exp_name + '.tgt'
        self.tar_out_path = os.path.join(outputs_dir, self.tar_out_name)

    def __compile_new_model(self, pad_id):
        optimizer = keras.optimizers.Adam(lr=self.args.lr, beta_1=0.9, beta_2=0.999)
        _model = self.uni_transformer_model.get_model(pad_id)
        _model.compile(
                       optimizer=optimizer,
                       loss = keras.losses.sparse_categorical_crossentropy,
                      )
        return _model

    def train(self):
        ds = DataSet(self.args)
        print('*' * 100)
        print('train sample number: ', ds.train_sample_num)
        print('valid sample number: ', ds.valid_sample_num)
        print('test sample number: ', ds.test_sample_num)
        print('*' * 100)

        train_generator = ds.data_generator('train', 'universal_transformer')

        valid_generator = ds.data_generator('valid', 'universal_transformer')

        if os.path.exists(self.model_path):
        #if os.path.exists(self.model_weights_path):
            print('Loading model from: %s' % self.model_path)
            custom_dict = get_custom_objects()
            model = load_model(self.model_path, custom_objects=custom_dict)
            #model = self.__compile_new_model(ds.pad_id)
            #model.load_weights(self.model_weights_path)
        else:
            print('Compile new model...')
            model = self.__compile_new_model(ds.pad_id)

        #model.summary()
        #plot_model(model, to_file='model_structure.png',show_shapes=True)

        verbose = 1
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.args.early_stop_patience, verbose=verbose)
        ckpt_name = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        ckpt_path = os.path.join(self.exp_dir, ckpt_name)
        checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')

        callback_list = [earlystopper, checkpoint]
    
        hist = model.fit_generator(
                        generator=train_generator, 
                        steps_per_epoch=(ds.train_sample_num//self.args.batch_size),
                        epochs=self.args.epochs,
                        callbacks=callback_list, 
                        validation_data=valid_generator,
                        validation_steps=(ds.valid_sample_num//self.args.batch_size),
                       )
        with open(self.history_path,'w') as f:
            f.write(str(hist.history))

        #model.save_weights(self.model_weights_path)
        model.save(self.model_path)
        #plot_model(model, to_file='model_structure.png',show_shapes=True) 

    def test(self):
        ds = DataSet(args)
        test_generator = ds.data_generator('test', 'universal_transformer')
                            #max_src_len=self.args.src_seq_length, 
                            #max_tar_len=self.args.tar_seq_length, 
                            #)
        # load_model
        print('Loading model from: %s' % self.model_path)
        custom_dict = get_custom_objects()
        #model = self.__compile_new_model(ds.pad_id)
        #model.load_weights(self.model_weights_path)
        model = load_model(self.model_path, custom_objects=custom_dict)

        src_outobj = open(self.src_out_path, 'w')
        pred_outobj = open(self.pred_out_path, 'w')
        tar_outobj = open(self.tar_out_path, 'w')
    
        for batch, ([src_input, tar_input], tar_loss_input) in enumerate(test_generator):
            if batch > (ds.test_sample_num // self.args.batch_size):
                # finish all of the prediction
                break
            print('Current batch: {}/{}. '.format(batch, ds.test_sample_num // self.args.batch_size))
            cur_batch_size = tar_input.shape[0]
            tar_length = tar_input.shape[1]

            results = np.zeros_like(tar_input)
            results[:, 0] = ds.start_id

            for t in range(1, tar_length):
                preds = model.predict([src_input, np.asarray(results)]) # shape: (batch_size, tar_length, vocab_size)
                pred_id = np.argmax(preds, axis=-1)
                results[:, t] = pred_id[:, t-1]

            def output_results(outputs, outobj):
                for result in outputs:
                    seq = []
                    for _id in result:
                        _id = int(_id)
                        if _id == ds.end_id:
                            break
                        if _id != ds.pad_id and _id != ds.start_id:
                            seq.append(ds.tar_id_tokens.get(_id, config.UNK_TOKEN))
                    write_line = ' '.join(seq)
                    write_line = write_line + '\n'
                    outobj.write(write_line)
    
            output_results(results, pred_outobj)
            output_results(src_input, src_outobj)
            output_results(tar_input, tar_outobj)
    
        src_outobj.close()
        pred_outobj.close()
        tar_outobj.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    universal_transformer_add_arguments(parser)
    args = parser.parse_args()
    print(args)

    trans = UniversalTransformer(args)
    trans.train()
    trans.test()


