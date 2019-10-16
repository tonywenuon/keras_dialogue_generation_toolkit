import os, sys, time, math

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import tensorflow as tf
import keras
import argparse
import numpy as np
from copy import deepcopy
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import get_custom_objects
from models.transformer_ed import TEDModel
from commonly_used_code.helper_fn import Hypothesis
from commonly_used_code import helper_fn, config
from run_script.args_parser import ted_add_arguments
from vspgt_data_reader import DataSet
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class TransformerED:
    def __init__(self, args):
        # real Transformer model architecture
        self.transformer_model = TEDModel(args=args,
                                                  transformer_dropout=0.05,
                                                  embedding_dropout=0.05,
                                                  use_same_embedding=True,
                                                  )
        self.args = args
        exp_name = args.data_set + '_' + args.exp_name

        # create experiment dir
        self.exp_dir= os.path.join(args.checkpoints_dir, exp_name)
        helper_fn.makedirs(self.exp_dir)
        hist_name = exp_name + '.hist'
        model_name = exp_name + '_final_model.h5'

        self.history_path = os.path.join(self.exp_dir, hist_name)
        self.model_path = os.path.join(self.exp_dir, model_name)
        
        outputs_dir = args.outputs_dir
        helper_fn.makedirs(outputs_dir)
        self.src_out_name = exp_name + '.src'
        self.src_out_path = os.path.join(outputs_dir, self.src_out_name)
        self.pred_out_name = exp_name + '.pred'
        self.pred_out_path = os.path.join(outputs_dir, self.pred_out_name)
        self.tar_out_name = exp_name + '.tgt'
        self.tar_out_path = os.path.join(outputs_dir, self.tar_out_name)

    def train(self):
        ds = DataSet(self.args)
        print('*' * 100)
        print('train sample number: ', ds.train_sample_num)
        print('valid sample number: ', ds.valid_sample_num)
        print('test sample number: ', ds.test_sample_num)
        print('*' * 100)

        train_generator = ds.data_generator('train', 'ted')
        valid_generator = ds.data_generator('valid', 'ted')

        def compile_new_model():
            _model = self.transformer_model.get_model(ds.pad_id)
            _model.compile(
                           optimizer=keras.optimizers.Adam(lr=self.args.lr),
                           loss = keras.losses.sparse_categorical_crossentropy,
                          )
            return _model

        if os.path.exists(self.model_path):
            print('Loading model from: %s' % self.model_path)
            custom_dict = get_custom_objects()
            model = load_model(self.model_path, custom_objects=custom_dict)
        else:
            print('Compile new model...')
            model = compile_new_model()

        model.summary()
        plot_model(model, to_file='model_structure.png',show_shapes=True)

        verbose = 1
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.args.early_stop_patience, verbose=verbose)
        ckpt_name = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        ckpt_path = os.path.join(self.exp_dir, ckpt_name)
        checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=verbose, save_weights_only=True, save_best_only=True, mode='min')
        lrate = keras.callbacks.ReduceLROnPlateau(
                                     monitor='val_loss', 
                                     factor=0.5, 
                                     patience=self.args.lr_decay_patience, 
                                     verbose=verbose, 
                                     mode='auto', 
                                     min_delta=0.0001, 
                                     cooldown=0, 
                                     min_lr=self.args.lr_min,
                                     )

        callback_list = [earlystopper, checkpoint, lrate]
        #callback_list = [earlystopper, lrate]
    
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

        model.save(self.model_path)
        #model.save_weights(self.model_path)
        #plot_model(model, to_file='model_structure.png',show_shapes=True) 

    def test(self):
        ds = DataSet(args)
        test_generator = ds.data_generator('test', 'ted')

        def compile_new_model():
            _model = self.transformer_model.get_model(ds.pad_id)
            _model.compile(
                           optimizer=keras.optimizers.Adam(lr=self.args.lr),
                           loss = keras.losses.sparse_categorical_crossentropy,
                          )
            return _model

        # load_model
        print('Loading model from: %s' % self.model_path)
        custom_dict = get_custom_objects()
        model = load_model(self.model_path, custom_objects=custom_dict)
        #model = compile_new_model()
        #model.load_weights(self.model_path)


        src_outobj = open(self.src_out_path, 'w')
        pred_outobj = open(self.pred_out_path, 'w')
        tar_outobj = open(self.tar_out_path, 'w')
    
        for batch, ([src_input, tar_input, facts_input], tar_loss_input) in enumerate(test_generator):
            if batch >= (ds.test_sample_num // self.args.batch_size):
                # finish all of the prediction
                break
            print('Current batch: {}/{}. '.format(batch, ds.test_sample_num // self.args.batch_size))
            cur_batch_size = tar_input.shape[0]
            tar_length = tar_input.shape[1]

            results = np.zeros_like(tar_input)
            results[:, 0] = ds.start_id
            for i in range(1, tar_length):
                results[:, i] = ds.pad_id

            for t in range(1, tar_length):
                preds = model.predict([src_input, np.asarray(results), facts_input]) # shape: (batch_size, tar_length, vocab_size)
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
                    outobj.flush()
    
            output_results(results, pred_outobj)
            output_results(src_input, src_outobj)
            output_results(tar_input, tar_outobj)
    
        src_outobj.close()
        pred_outobj.close()
        tar_outobj.close()
        print(self.pred_out_path)

    def beam_search_test(self):
        beam_size = self.args.beam_size
        ds = DataSet(args)
        test_generator = ds.data_generator('test', 'ted')

        def sort_for_each_hyp(hyps, rank_index):
            """Return a list of Hypothesis objects, sorted by descending average log probability"""
            return sorted(hyps, key=lambda h: h.avg_prob[rank_index], reverse=True)

        def get_new_hyps(all_hyps):
            hyp = all_hyps[0]
            batch_size = hyp.batch_size
            tar_len = hyp.tar_len

            new_hyps = []
            for i in range(beam_size):
                hyp = Hypothesis(batch_size, tar_length, ds.start_id, ds.end_id)
                new_hyps.append(hyp)
            for i in range(batch_size):
                # rank based on each sample's probs
                sorted_hyps = sort_for_each_hyp(all_hyps, i)
                for j in range(beam_size):
                    hyp = sorted_hyps[j]
                    new_hyps[j].res_ids[i] = hyp.res_ids[i]
                    new_hyps[j].pred_ids[i] = hyp.pred_ids[i]
                    new_hyps[j].probs[i] = hyp.probs[i]
            return new_hyps

        def update_hyps(all_hyps):
            # all_hyps: beam_size * beam_size current step hyps. 
            new_hyps = get_new_hyps(all_hyps)
            return new_hyps

        def get_final_results(hyps):
            hyp = hyps[0]
            batch_size = hyp.batch_size
            tar_len = hyp.tar_len

            final_hyp = Hypothesis(batch_size, tar_length, ds.start_id, ds.end_id)
            for i in range(batch_size):
                # rank based on each sample's probs
                sorted_hyps = sort_for_each_hyp(hyps, i)
                hyp = sorted_hyps[0]
                final_hyp.res_ids[i] = hyp.res_ids[i]
                final_hyp.pred_ids[i] = hyp.pred_ids[i]
                final_hyp.probs[i] = hyp.probs[i]
            res = np.asarray(final_hyp.res_ids)
            return res

        # load_model
        def compile_new_model():
            _model = self.transformer_model.get_model(ds.pad_id)
            _model.compile(
                           optimizer=keras.optimizers.Adam(lr=self.args.lr),
                           loss = keras.losses.sparse_categorical_crossentropy,
                          )
            return _model

        # load_model
        print('Loading model from: %s' % self.model_path)
        custom_dict = get_custom_objects()
        model = load_model(self.model_path, custom_objects=custom_dict)
        #model = compile_new_model()
        #model.load_weights(self.model_path)

        src_outobj = open(self.src_out_path, 'w')
        pred_outobj = open(self.pred_out_path, 'w')
        tar_outobj = open(self.tar_out_path, 'w')

        for batch_index, ([src_input, tar_input, facts_input], tar_loss_input) in enumerate(test_generator):
            if batch_index > (ds.test_sample_num // self.args.batch_size):
                # finish all of the prediction
                break

            print('Current batch: {}/{}. '.format(batch_index, ds.test_sample_num // self.args.batch_size))
            cur_batch_size = tar_input.shape[0]
            tar_length = tar_input.shape[1]
            hyps = []
            for i in range(beam_size):
                hyp = Hypothesis(cur_batch_size, tar_length, ds.start_id, ds.end_id)
                hyps.append(hyp)

            for t in range(1, tar_length):
                # iterate each sample
                # collect all hyps, basically, it's beam_size * beam_size
                all_hyps = []
                for i in range(beam_size):
                    cur_hyp = hyps[i]
                    results = cur_hyp.get_predictable_vars(ds.pad_id)
                    # bs, tar_len, 60000
                    preds = model.predict([src_input, np.asarray(results), facts_input]) # shape: (batch_size, tar_length, vocab_size)
                        
                    # get the current step prediction
                    cur_preds = preds[:, t - 1]

                    # tar_len, 60000
                    top_indices = np.argsort(cur_preds)
                    top_indices = top_indices[:, -beam_size:] # the largest one is at the end
                        
                    top_logits = []
                    for sample_index, sample_logits in enumerate(cur_preds):
                        logits = []
                        for beam_index in range(beam_size):
                            logit = sample_logits[top_indices[sample_index][beam_index]]
                            logits.append(logit)
                        top_logits.append(logits)
                    top_logits = np.asarray(top_logits)

                    # iterate each new prediction
                    for j in range(beam_size-1, -1, -1):
                        next_hyp = deepcopy(cur_hyp)
                        # bs, 1
                        top_index = top_indices[:, j]
                        top_logit = top_logits[:, j]

                        for bs_idx, _id in enumerate(top_index):
                            next_hyp.res_ids[bs_idx].append(_id)
                            prob = top_logit[bs_idx]
                            next_hyp.probs[bs_idx].append(prob)

                            # get OOV id
                            token = ds.tar_id_tokens.get(int(_id), config.UNK_TOKEN)
                            if token == config.UNK_TOKEN:
                                cur_pred_id = ds.unk_id
                            else:
                                cur_pred_id = _id
                            next_hyp.pred_ids[bs_idx].append(cur_pred_id)

                        all_hyps.append(next_hyp)

                    # if it is the first step, only predict once
                    if t == 1:
                        break
                hyps = update_hyps(all_hyps)
            final_results = get_final_results(hyps)

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
                    outobj.flush()
    
            output_results(results, pred_outobj)
            output_results(src_input, src_outobj)
            output_results(tar_input, tar_outobj)
    
        src_outobj.close()
        pred_outobj.close()
        tar_outobj.close()

        preds = []
        with open(self.pred_out_path) as f:
            for line in f:
                preds.append(line.strip())
        tgts = []
        with open(self.tar_out_path) as f:
            for line in f:
                tgts.append(line.strip())
        exact_scores, f1_scores = metrics.get_raw_scores(tgts, preds)
        em_f1 = metrics.make_eval_dict(exact_scores, f1_scores)
        print(em_f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ted_add_arguments(parser)
    args = parser.parse_args()
    print(args)

    trans = TransformerED(args)
    trans.train()
    trans.test()
#    trans.beam_search_test()




