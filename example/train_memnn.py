import os, sys, time, math

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import tensorflow as tf
import argparse
import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from models.memnn import MemNNModel, get_custom_objects
from commonly_used_code import helper_fn, config
from run_script.args_parser import memnn_add_arguments
from data_reader import DataSet
import keras.backend.tensorflow_backend as KTF

#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class MemNN:
    def __init__(self, args):
        # real MemNN model architecture
        self.memnn_model = MemNNModel(args)
        self.args = args
        exp_name = args.data_set + '_' + args.exp_name

        # create experiment dir
        self.exp_dir= os.path.join(args.checkpoints_dir, exp_name)
        helper_fn.makedirs(self.exp_dir)
        hist_name = exp_name + '.hist'
        model_name = exp_name + '_final_model.h5'
        encoder_model_name = exp_name + '_encoder_model.h5'
        decoder_model_name = exp_name + '_decoder_model.h5'

        self.history_path = os.path.join(self.exp_dir, hist_name)
        self.model_path = os.path.join(self.exp_dir, model_name)
        self.encoder_model_path = os.path.join(self.exp_dir, encoder_model_name)
        self.decoder_model_path = os.path.join(self.exp_dir, decoder_model_name)
        
        outputs_dir = args.outputs_dir
        helper_fn.makedirs(outputs_dir)
        self.src_out_name = exp_name + '.src'
        self.src_out_path = os.path.join(outputs_dir, self.src_out_name)
        self.pred_out_name = exp_name + '.pred'
        self.pred_out_path = os.path.join(outputs_dir, self.pred_out_name)
        self.tar_out_name = exp_name + '.tgt'
        self.tar_out_path = os.path.join(outputs_dir, self.tar_out_name)

    def train(self, tag):
        ds = DataSet(self.args)
        _, train_src_ids, train_tar_ids, train_tar_loss_ids, _, train_facts_ids = \
        ds.read_file('train', 
                     max_src_len=self.args.src_seq_length, 
                     max_tar_len=self.args.tar_seq_length, 
                     max_fact_len=self.args.fact_seq_length, 
                     get_fact=True, 
                     get_one_hot=False)
    
        _, valid_src_ids, valid_tar_ids, valid_tar_loss_ids, _, valid_facts_ids = \
        ds.read_file('valid', 
                     max_src_len=self.args.src_seq_length, 
                     max_tar_len=self.args.tar_seq_length, 
                     max_fact_len=self.args.fact_seq_length, 
                     get_fact=True, 
                     get_one_hot=False)
    
        
        if tag == 'train':
            model, encoder_model, decoder_model = MemNNModel(args).get_model()
        elif tag == 'retrain':
            custom_dict = get_custom_objects()
            model = load_model(self.model_path, custom_objects=custom_dict, compile=False)
            encoder_model = load_model(self.encoder_model_path)
            decoder_model = load_model(self.decoder_model_path)
        # When using sparse_categorical_crossentropy your labels should be of shape (batch_size, seq_length, 1) instead of simply (batch_size, seq_length).
        opt = tf.keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        model.compile(optimizer=opt,
                      loss=tf.keras.losses.sparse_categorical_crossentropy, 
        )
    
        verbose = 1
        earlystopper = EarlyStopping(monitor='val_loss', patience=args.early_stop_patience, verbose=verbose)
        ckpt_name = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        ckpt_path = os.path.join(self.exp_dir, ckpt_name)
        checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')
        lrate = ReduceLROnPlateau(
                                  monitor='val_loss', 
                                  factor=0.5, 
                                  patience=args.lr_decay_patience, 
                                  verbose=verbose, 
                                  mode='auto', 
                                  epsilon=0.0001, 
                                  cooldown=0, 
                                  min_lr=args.lr_min
                                  )
        #callback_list = [earlystopper, checkpoint, lrate]
        callback_list = [lrate]
    
        inp_y=np.expand_dims(np.asarray(train_tar_loss_ids), axis=-1)
        inp_valid_y=np.expand_dims(np.asarray(valid_tar_loss_ids), axis=-1)
        hist = model.fit(x=[
                            np.asarray(train_src_ids),
                            np.asarray(train_facts_ids),
                            np.asarray(train_tar_ids),
                           ],
                         y=inp_y,
                         epochs=args.epochs,
                         batch_size=args.batch_size,
                         callbacks=callback_list, 
                         validation_data=([
                                           np.asarray(valid_src_ids), 
                                           np.asarray(valid_facts_ids),
                                           np.asarray(valid_tar_ids),
                                          ], 
                                          inp_valid_y)
                         )
        with open(self.history_path,'w') as f:
            f.write(str(hist.history))
        # there is something wrong with Keras to save model and load_model. non-serialized problem
        model.save(self.model_path)

        return model

    def test(self, model):
        # load_model
        ds = DataSet(args)
        _, test_src_ids, test_tar_ids, test_tar_loss_ids, _, test_facts_ids = \
        ds.read_file('test', 
                     max_src_len=self.args.src_seq_length, 
                     max_tar_len=self.args.tar_seq_length, 
                     max_fact_len=self.args.fact_seq_length, 
                     get_fact=True, 
                     get_one_hot=False)

        src_outobj = open(self.src_out_path, 'w')
        pred_outobj = open(self.pred_out_path, 'w')
        tar_outobj = open(self.tar_out_path, 'w')
    
        def __get_batch():
            batch_src = []
            batch_facts = []
            batch_tar = []
            for (src_input, facts_input, tar_input) in zip(test_src_ids, test_facts_ids, test_tar_ids):
                batch_src.append(src_input)
                batch_facts.append(facts_input)
                batch_tar.append(tar_input)
                if len(batch_src) == self.args.batch_size:
                    res = (np.asarray(batch_src), np.asarray(batch_facts), np.asarray(batch_tar))
                    batch_src = []
                    batch_facts = []
                    batch_tar = []
                    yield res[0], res[1], res[2]
            yield np.asarray(batch_src), np.asarray(batch_facts), np.asarray(batch_tar)

        for (batch, (src_input, facts_input, tar_input)) in enumerate(__get_batch()):
            if batch >= (ds.test_sample_num // self.args.batch_size):
                # finish all of the prediction
                break
            print('Current batch: {}/{}. '.format(batch, len(test_src_ids) // self.args.batch_size))
            cur_batch_size = tar_input.shape[0]
            tar_length = tar_input.shape[1]

            results = []
            results = np.zeros((cur_batch_size, tar_length), dtype='int32')
            results[:, 0] = ds.start_id

            for t in range(1, tar_length):
                preds = model.predict([src_input, facts_input, results]) # shape: (batch_size, tar_length, vocab_size)
                pred_id = np.argmax(preds, axis=-1)
                results[:, t] = pred_id[:, t - 1]

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
    memnn_add_arguments(parser)
    args = parser.parse_args()

    # tag can be train or retrain
    tag = 'train'
    mn = MemNN(args)
    model = mn.train(tag)
    mn.test(model)




