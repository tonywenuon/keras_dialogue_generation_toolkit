
import os, sys, time, math

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import re
import tensorflow as tf
tf.enable_eager_execution()
import argparse
import numpy as np
from models.seq2seq_attn import Encoder, Decoder
from commonly_used_code import helper_fn, config
from run_script.args_parser import seq2seq_attn_add_arguments
from data_reader  import DataSet
import keras.backend.tensorflow_backend as KTF

#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':0})))  
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Seq2Seq:
    def __init__(self, args):
        self.args = args
        exp_name = args.data_set + '_' + args.exp_name

        self.checkpoints_dir = os.path.join(args.checkpoints_dir, exp_name)
        helper_fn.makedirs(self.checkpoints_dir)
        self.checkpoint_prefix = os.path.join(self.checkpoints_dir, 'ckpt')
        self.tensorboard_dir = args.tensorboard_dir
        helper_fn.makedirs(self.tensorboard_dir)
        
        outputs_dir = args.outputs_dir
        helper_fn.makedirs(outputs_dir)
        self.src_out_name = exp_name + '.src'
        self.src_out_path = os.path.join(outputs_dir, self.src_out_name)
        self.pred_out_name = exp_name + '.pred'
        self.pred_out_path = os.path.join(outputs_dir, self.pred_out_name)
        self.tar_out_name = exp_name + '.tgt'
        self.tar_out_path = os.path.join(outputs_dir, self.tar_out_name)

    
    def __loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def train(self, _type):
        ds = DataSet(args)
        _, train_src_ids, train_tar_ids, _, _, _ = \
        ds.read_file('train', 
                     max_src_len=self.args.src_seq_length, 
                     max_tar_len=self.args.tar_seq_length, 
                    )

        dataset = tf.data.Dataset.from_tensor_slices((train_src_ids, train_tar_ids))
        dataset = dataset.batch(self.args.batch_size)
        n_batch = len(train_src_ids) // self.args.batch_size

        _, valid_src_ids, valid_tar_ids, _, _, _ = \
        ds.read_file('valid', 
                     max_src_len=self.args.src_seq_length, 
                     max_tar_len=self.args.tar_seq_length, 
                    )

        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_src_ids, valid_tar_ids))
        valid_dataset = valid_dataset.batch(self.args.batch_size)

    
        encoder = Encoder(ds.src_vocab_size, self.args.embedding_dim, self.args.hidden_dim, self.args.batch_size)
        decoder = Decoder(ds.tar_vocab_size, self.args.embedding_dim, self.args.hidden_dim, self.args.batch_size)
        
        optimizer = tf.train.AdamOptimizer()
        checkpoint = tf.contrib.eager.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        if _type == 'retrain':
            checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
    
        min_valid_loss = math.inf
        improve_num = 0
        summary_writer = tf.contrib.summary.create_file_writer(self.tensorboard_dir)
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            for epoch in range(self.args.epochs):
                start = time.time()
                total_loss = 0
                for (batch, (src_input, tar_input)) in enumerate(dataset):
                    loss = 0
                    hidden_state = encoder.initial_hidden_state(src_input.shape[0])
                    with tf.GradientTape() as tape:
                        enc_output, enc_state = encoder(src_input, hidden_state)
                        dec_state = enc_state
    
                        dec_input = tf.keras.backend.expand_dims([ds.start_id] * src_input.shape[0], 1)
                        for t in range(1, tar_input.shape[1]):
                            # teacher - forcing. feeding the target as the next input
                            preds, dec_state, _ = decoder(dec_input, dec_state, enc_output)
                            loss += self.__loss_function(tar_input[:, t], preds)
                            dec_input = tf.keras.backend.expand_dims(tar_input[:, t], 1)

                    batch_loss = (loss / int(tar_input.shape[1]))
                    total_loss += batch_loss
                    variables = encoder.variables + decoder.variables
                    gradients = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(gradients, variables))
    
                    if batch % self.args.display_step == 0:
                        print('Epoch {}/{}, Batch {}/{}, Batch Loss {:.4f}'.format(epoch + 1,
                                                                       self.args.epochs,
                                                                       batch, 
                                                                       n_batch,
                                                                       batch_loss.numpy()))
                    tf.contrib.summary.scalar("total_loss", total_loss)

                valid_total_loss = self.valid(valid_dataset, encoder, decoder, ds.start_id)
                print('Epoch {}, Train Loss {:.4f}, Valid Loss {:.4f}'.format(epoch + 1, 
                                                                              total_loss / n_batch, 
                                                                              valid_total_loss))
                if valid_total_loss < min_valid_loss:
                    improve_num = 0
                    print('Valid loss improves from {}, to {}'.format(min_valid_loss, valid_total_loss))
                    min_valid_loss = valid_total_loss
                    checkpoint.save(file_prefix=self.checkpoint_prefix)
                elif valid_total_loss >= min_valid_loss:
                    improve_num += 1
                    print('Valid loss did not improve from {}'.format(min_valid_loss))
                    if improve_num >= self.args.early_stop_patience:
                        break
                print('Time taken for epoch {}: {} sec \n'.format(epoch + 1, 
                                                                  time.time() - start))
        checkpoint.save(file_prefix=self.checkpoint_prefix)


    def valid(self, valid_dataset, encoder, decoder, start_id):
        total_loss = 0
        for (batch, (src_input, tar_input)) in enumerate(valid_dataset):
            loss = 0
            hidden_state = encoder.initial_hidden_state(src_input.shape[0])
            enc_output, enc_state = encoder(src_input, hidden_state)
            dec_state = enc_state
    
            dec_input = tf.keras.backend.expand_dims([start_id] * src_input.shape[0], 1)
            for t in range(1, tar_input.shape[1]):
                preds, dec_state, _ = decoder(dec_input, dec_state, enc_output)
                loss += self.__loss_function(tar_input[:, t], preds)
    
                pred_id = tf.keras.backend.argmax(preds, axis=1).numpy()
                dec_input = tf.keras.backend.expand_dims(tar_input[:, t], 1)

            batch_loss = (loss / int(tar_input.shape[1]))
            total_loss += batch_loss

        return total_loss

    
    def test(self):
        ds = DataSet(args)
        indexes, test_src_ids, test_tar_ids, _, _, _ = \
        ds.read_file('test', 
                     max_src_len=self.args.src_seq_length, 
                     max_tar_len=self.args.tar_seq_length, 
                    )

        dataset = tf.data.Dataset.from_tensor_slices((indexes, test_src_ids, test_tar_ids))
        dataset = dataset.batch(self.args.batch_size)
        n_batch = len(test_src_ids) // self.args.batch_size
        print('*' * 100)
        print('Test set size: %d' % len(test_src_ids))
        print('*' * 100)
    
        encoder = Encoder(ds.src_vocab_size, self.args.embedding_dim, self.args.hidden_dim, self.args.batch_size)
        decoder = Decoder(ds.tar_vocab_size, self.args.embedding_dim, self.args.hidden_dim, self.args.batch_size)
        optimizer = tf.train.AdamOptimizer()
        checkpoint = tf.contrib.eager.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
    
        src_outobj = open(self.src_out_path, 'w')
        pred_outobj = open(self.pred_out_path, 'w')
        tar_outobj = open(self.tar_out_path, 'w')
    
        for (batch, (index_input, src_input, tar_input)) in enumerate(dataset):
            print('Current batch {}/{}. '.format(batch, n_batch))
            hidden_state = encoder.initial_hidden_state(src_input.shape[0])
            enc_output, enc_state = encoder(src_input, hidden_state)
            dec_state = enc_state
    
            results = np.zeros((tar_input.shape[0], tar_input.shape[1]))
            dec_input = tf.keras.backend.expand_dims([ds.start_id] * src_input.shape[0], 1)
            for t in range(1, tar_input.shape[1]):
                preds, dec_state, _ = decoder(dec_input, dec_state, enc_output)
                pred_id = tf.keras.backend.argmax(preds, axis=1).numpy()
                results[:, t] = pred_id
                dec_input = tf.reshape(pred_id, (-1, 1))

            def output_results(outputs, outobj, is_output_index=False):
                for idx, result in enumerate(outputs):
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

    # This is used for show attention for individual sample
    def decode(query):
        ds = DataSet(args)
    
        encoder = Encoder(ds.src_vocab_size, self.args.embedding_dim, self.args.hidden_dim, self.args.batch_size)
        decoder = Decoder(ds.tar_vocab_size, self.args.embedding_dim, self.args.hidden_dim, self.args.batch_size)
        optimizer = tf.train.AdamOptimizer()
        checkpoint = tf.contrib.eager.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))

        src_pad_length = self.args.src_seq_length
        tar_pad_length = self.args.tar_seq_length
        attn_plot = np.zeros((tar_pad_length, src_pad_length))
    
        query = tf.keras.backend.expand_dims(query, 0)
        hidden_state = encoder.initial_hidden_state(query.shape[0])
        enc_output, enc_state = encoder(query, hidden_state)
        dec_state = enc_state
    
        results = []
        dec_input = tf.keras.bakend.expand_dims([ds.start_id], 0)
        for t in range(1, tar_pad_length):
            preds, dec_state, attn_weights = decoder(dec_input, dec_state, enc_output)
    
            # store the attention weights to plot later
            attn_weights = tf.keras.backend.reshape(attn_weights, (-1, ))
            attn_plot[t] = attn_weights.numpy()
    
            pred_id = tf.keras.backend.argmax(preds[0]).numpy()
            if pred_id == ds.end_id:
                break
            if pred_id != ds.pad_id and pred_id != ds.start_id:
                results.append(ds.tar_id_tokens.get(pred_id, config.UNK_TOKEN))
   
            dec_input = tf.keras.backend.expand_dims([pred_id], 0)
        return results, attn_plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    seq2seq_attn_add_arguments(parser)
    args = parser.parse_args()

    s2s = Seq2Seq(args)
    # tag can be: train or retrain
    tag = 'train'
    s2s.train(tag)
    s2s.test()



