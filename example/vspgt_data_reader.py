
import sys, os
project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import argparse
import numpy as np
from copy import deepcopy
from sklearn.utils import shuffle
from typing import Callable, Optional, Sequence, Iterable
from commonly_used_code import config, helper_fn

class Batch:
    def __init__(self):
        self.init_all()

    def init_all(self):
        self.init_oovs()
        self.init_src_tar()
        self.init_fact()
        self.init_conv()

    def init_oovs(self):
        # each sample has an oov ids and its max oov lens
        # oov dict
        self.oov_ids_dict = []

    def init_src_tar(self):
        # normal ids
        self.src_ids = []
        self.tar_ids = []
        self.tar_loss_ids = []

        # for multi-task, take the first fact as the answer
        self.fact_tar_ids = []
        self.fact_tar_loss_ids = []

        # OOV expand for each data set
        self.src_ids_exp = []
        self.tar_ids_exp = []
        self.tar_loss_ids_exp = []

    def init_fact(self):
        self.fact_ids = []
        self.fact_ids_exp = []

    def init_conv(self):
        self.conv_ids = []
        self.conv_ids_exp = []

    def np_format(self):
        self.src_ids = np.asarray(self.src_ids)
        self.tar_ids = np.asarray(self.tar_ids)
        self.tar_loss_ids = np.asarray(self.tar_loss_ids)
        self.tar_loss_ids = np.reshape(self.tar_loss_ids, (self.tar_loss_ids.shape[0], self.tar_loss_ids.shape[1], 1))
        self.fact_ids = np.asarray(self.fact_ids)
        if len(self.conv_ids) != 0:
            self.conv_ids = np.asarray(self.conv_ids)

        # used for multi-task
        self.fact_tar_ids = np.asarray(self.fact_tar_ids)
        self.fact_tar_loss_ids = np.asarray(self.fact_tar_loss_ids)
        self.fact_tar_loss_ids = np.reshape(self.fact_tar_loss_ids, (self.fact_tar_loss_ids.shape[0], self.fact_tar_loss_ids.shape[1], 1))

        self.src_ids_exp = np.asarray(self.src_ids_exp)
        self.tar_ids_exp = np.asarray(self.tar_ids_exp)
        self.tar_loss_ids_exp = np.asarray(self.tar_loss_ids_exp)
        self.tar_loss_ids_exp = np.reshape(self.tar_loss_ids_exp, (self.tar_loss_ids_exp.shape[0], self.tar_loss_ids_exp.shape[1], 1))
        self.fact_ids_exp = np.asarray(self.fact_ids_exp)

        if len(self.conv_ids_exp) != 0:
            self.conv_ids_exp = np.asarray(self.conv_ids_exp)

    def batch_shuffle(self, have_conv=None):
        if have_conv == None:
            self.src_ids, self.tar_ids, self.tar_loss_ids, self.fact_ids, self.fact_tar_ids, self.fact_tar_loss_ids, \
                self.src_ids_exp, self.tar_ids_exp, self.tar_loss_ids_exp, self.fact_ids_exp = \
                    shuffle( self.src_ids, self.tar_ids, self.tar_loss_ids, self.fact_ids, self.fact_tar_ids, self.fact_tar_loss_ids, \
                        self.src_ids_exp, self.tar_ids_exp, self.tar_loss_ids_exp, self.fact_ids_exp)
        else:
            self.src_ids, self.tar_ids, self.tar_loss_ids, self.conv_ids, self.fact_ids, self.fact_tar_ids, self.fact_tar_loss_ids, \
                self.src_ids_exp, self.tar_ids_exp, self.tar_loss_ids_exp, self.fact_ids_exp = \
                    shuffle( self.src_ids, self.tar_ids, self.tar_loss_ids, self.conv_ids, self.fact_ids, self.fact_tar_ids, self.fact_tar_loss_ids, \
                        self.src_ids_exp, self.tar_ids_exp, self.tar_loss_ids_exp, self.fact_ids_exp)

class DataSet:
    def __init__(self, args):
        self.args = args
        self.__set_file_path()

        # get global token and ids 
        self.src_token_ids, self.src_id_tokens, self.src_vocab_size = self.__read_global_ids(self.src_global_token_path)
        self.tar_token_ids, self.tar_id_tokens, self.tar_vocab_size = self.__read_global_ids(self.tar_global_token_path)
        self.train_sample_num = 0
        self.valid_sample_num = 0
        self.test_sample_num = 0

        # just set the max fact and conv length the same as source
        self.max_src_len = args.src_seq_length
        self.max_tar_len = args.tar_seq_length
        if not args.fact_seq_length:
            self.max_fact_len = args.src_seq_length
        else:
            print('This is fact length')
            self.max_fact_len = args.fact_seq_length
        #if not args.conv_seq_length:
        self.max_conv_len = args.conv_seq_length
        #else:
        #    self.max_conv_len = args.conv_seq_length
        self.src_len = self.max_src_len - config.src_reserved_pos
        self.tar_len = self.max_tar_len - config.tar_reserved_pos

        self.pad_fact_seqs = helper_fn.pad_with_pad([self.pad_id], self.max_fact_len, self.pad_id)
        self.pad_conv_seqs = helper_fn.pad_with_pad([self.pad_id], self.max_conv_len, self.pad_id)

        self.__get_sample_numbers()

    def __set_file_path(self):
        if self.args.data_set == 'example_data':
            self.train_set_path = config.example_data_train_qa_path
            self.train_conv_path = config.example_data_train_conv_path
            self.train_sent_fact_path = config.example_data_train_sent_fact_path
            self.valid_set_path = config.example_data_valid_qa_path
            self.valid_conv_path = config.example_data_valid_conv_path
            self.valid_sent_fact_path = config.example_data_valid_sent_fact_path
            self.test_set_path = config.example_data_test_qa_path
            self.test_conv_path = config.example_data_test_conv_path
            self.test_sent_fact_path = config.example_data_test_sent_fact_path
            self.src_global_token_path = config.example_data_global_token_path
            self.tar_global_token_path = config.example_data_global_token_path

    def __get_sample_numbers(self):
        print('Getting total samples numbers...')
        with open(self.train_set_path) as f:
            for line in f:
                self.train_sample_num += 1
        with open(self.valid_set_path) as f:
            for line in f:
                self.valid_sample_num += 1
        with open(self.test_set_path) as f:
            for line in f:
                self.test_sample_num += 1
                
    def _line2ids(self, line, max_len, oov_token2id=None):
        seq = []
        seq_exp = []
        for token in line.strip().split(' '):
            _id = self.src_token_ids.get(token, self.unk_id)
            seq.append(_id)
            _id_exp = _id
            if _id == self.unk_id:
                if oov_token2id != None:
                    _id_exp = oov_token2id.get(token, self.unk_id)
            seq_exp.append(_id_exp)

        seq = seq[:max_len]
        seq_exp = seq_exp[:max_len]
        return seq, seq_exp

    def _get_oov_dict(self, qa_line, fact_line):
        oov_token2id = dict()
        oov_id2token = dict()
        oov_index = self.src_vocab_size
        # just calculate question, ignoring the target tokens
        ques_line = qa_line.strip().split('\t')[1]
        for token in ques_line.strip().split(' '):
            token = token.strip()
            _id = self.src_token_ids.get(token, self.unk_id)
            if _id == self.unk_id:
                if token not in oov_token2id:
                    oov_token2id[token] = oov_index
                    oov_id2token[oov_index] = token
                    oov_index += 1

        elems = fact_line.strip().split('\t')
        for fact in elems[1:self.args.fact_number + 1]:
            for token in fact.strip().split(' '): 
                token = token.strip()
                _id = self.src_token_ids.get(token, self.unk_id)
                if _id == self.unk_id:
                    if token not in oov_token2id:
                        oov_token2id[token] = oov_index
                        oov_id2token[oov_index] = token
                        oov_index += 1

        return oov_token2id, oov_id2token

    def _deal_qa_line(self, index, line, oov_token2id):
        elems = line.strip().split('\t')

        text = elems[1].strip()
        seq, seq_exp = self._line2ids(text, self.src_len, oov_token2id)
        src = helper_fn.pad_with_start_end(seq, self.max_src_len, self.start_id, self.end_id, self.pad_id)
        src_exp = helper_fn.pad_with_start_end(seq_exp, self.max_src_len, self.start_id, self.end_id, self.pad_id)

        # used for multi_task. If there is no fact, use src as the answer
        seq, seq_exp = self._line2ids(text, self.tar_len, oov_token2id)
        src_tar = helper_fn.pad_with_start(seq, self.max_tar_len, self.start_id, self.pad_id)
        src_tar_loss = helper_fn.pad_with_end(seq, self.max_tar_len, self.end_id, self.pad_id)

        text = elems[2].strip()
        seq, seq_exp = self._line2ids(text, self.tar_len, oov_token2id)
        tar = helper_fn.pad_with_start(seq, self.max_tar_len, self.start_id, self.pad_id)
        tar_exp = helper_fn.pad_with_start(seq_exp, self.max_tar_len, self.start_id, self.pad_id)

        tar_loss = helper_fn.pad_with_end(seq, self.max_tar_len, self.end_id, self.pad_id)
        tar_loss_exp = helper_fn.pad_with_end(seq_exp, self.max_tar_len, self.end_id, self.pad_id)

        return src, tar, tar_loss, src_exp, tar_exp, tar_loss_exp, src_tar, src_tar_loss

    def _deal_fact_line(self, index, line, oov_token2id):
        line = line.strip()
        cur_fact_ids = []
        cur_fact_ids_exp = []
        fact_tar = None
        fact_tar_loss = None
        elems = line.split('\t')
        no_fact_tag = False
        # if there is no fact, add pad sequence
        if elems[1] == config.NO_FACT or elems[1] == config.NO_CONTEXT:
            cur_fact_ids.append(self.pad_fact_seqs)
            cur_fact_ids_exp.append(self.pad_fact_seqs)
            no_fact_tag = True
        else:
            for index, text in enumerate(elems[1:]):
                seq, seq_exp = self._line2ids(text, self.max_fact_len, oov_token2id)
                new_seq = helper_fn.pad_with_pad(seq, self.max_fact_len, self.pad_id)
                cur_fact_ids.append(new_seq)
                new_seq_exp = helper_fn.pad_with_pad(seq_exp, self.max_fact_len, self.pad_id)
                cur_fact_ids_exp.append(new_seq_exp)
                if index == 0:
                    seq, seq_exp = self._line2ids(text, self.tar_len, oov_token2id)
                    fact_tar = helper_fn.pad_with_start(seq, self.max_tar_len, self.start_id, self.pad_id)
                    fact_tar_loss = helper_fn.pad_with_end(seq, self.max_tar_len, self.start_id, self.pad_id)

        # pad fact number
        cur_fact_ids = cur_fact_ids[:self.args.fact_number]
        cur_fact_ids_exp = cur_fact_ids_exp[:self.args.fact_number]

        cur_fact_ids = cur_fact_ids + [self.pad_fact_seqs] * (self.args.fact_number - len(cur_fact_ids))
        cur_fact_ids_exp = cur_fact_ids_exp + [self.pad_fact_seqs] * (self.args.fact_number - len(cur_fact_ids_exp))

        return no_fact_tag, cur_fact_ids, cur_fact_ids_exp, fact_tar, fact_tar_loss

    def _deal_conv_line(self, index, line, oov_token2id):
        line = line.strip()
        cur_conv_ids = []
        cur_conv_ids_exp = []
        conv_tar = None
        conv_tar_loss = None
        elems = line.split('\t')
        no_conv_tag = False
        # if there is no conv, add pad sequence
        if elems[1] == config.NO_CONTEXT:
            cur_conv_ids.append(self.pad_conv_seqs)
            cur_conv_ids_exp.append(self.pad_conv_seqs)
            no_conv_tag = True
        else:
            for index, text in enumerate(elems[1:]):
                seq, seq_exp = self._line2ids(text, self.max_conv_len, oov_token2id)
                new_seq = helper_fn.pad_with_pad(seq, self.max_conv_len, self.pad_id)
                cur_conv_ids.append(new_seq)
                new_seq_exp = helper_fn.pad_with_pad(seq_exp, self.max_conv_len, self.pad_id)
                cur_conv_ids_exp.append(new_seq_exp)
                if index == 0:
                    seq, seq_exp = self._line2ids(text, self.tar_len, oov_token2id)
                    conv_tar = helper_fn.pad_with_start(seq, self.max_tar_len, self.start_id, self.pad_id)
                    conv_tar_loss = helper_fn.pad_with_end(seq, self.max_tar_len, self.start_id, self.pad_id)

        # pad conv number
        cur_conv_ids = cur_conv_ids[:self.args.conv_number]
        cur_conv_ids_exp = cur_conv_ids_exp[:self.args.conv_number]

        cur_conv_ids = cur_conv_ids + [self.pad_conv_seqs] * (self.args.conv_number - len(cur_conv_ids))
        cur_conv_ids_exp = cur_conv_ids_exp + [self.pad_conv_seqs] * (self.args.conv_number - len(cur_conv_ids_exp))

        return no_conv_tag, cur_conv_ids, cur_conv_ids_exp, conv_tar, conv_tar_loss

    def _check_and_shuffle(self, _batch, have_conv=None):
        #print('in check and shuffle...')
        #print('src_ids len: ', len(_batch.src_ids))
        #print('tar_ids len: ', len(_batch.tar_ids))
        #print('tar_loss_ids len: ', len(_batch.tar_loss_ids))
        #print('fact_ids len: ', len(_batch.fact_ids))

        assert (len(_batch.src_ids) == len(_batch.tar_ids))
        assert (len(_batch.src_ids) == len(_batch.tar_loss_ids))
        assert (len(_batch.src_ids) == len(_batch.fact_ids))
        assert (len(_batch.src_ids) == len(_batch.fact_tar_ids))
        assert (len(_batch.src_ids_exp) == len(_batch.tar_ids_exp))
        assert (len(_batch.src_ids_exp) == len(_batch.tar_loss_ids_exp))
        assert (len(_batch.src_ids_exp) == len(_batch.fact_ids_exp))

        assert (len(_batch.src_ids) > 0)
        assert (len(_batch.tar_ids) > 0)
        assert (len(_batch.tar_loss_ids) > 0)
        assert (len(_batch.fact_ids) > 0)
        assert (len(_batch.fact_tar_ids) > 0)

        assert (len(_batch.src_ids_exp) > 0)
        assert (len(_batch.tar_ids_exp) > 0)
        assert (len(_batch.tar_loss_ids_exp) > 0)
        assert (len(_batch.fact_ids_exp) > 0)

        if have_conv != None:
            assert (len(_batch.src_ids) == len(_batch.conv_ids))

        return _batch.batch_shuffle(have_conv)

    def _fit_model(self, _batch, file_type, model_type):
        '''
        Please carefully choose the output type to fit with your model's inputs
        '''

        fit_output_loss = np.zeros((_batch.src_ids.shape[0], 1))

        no_fact_list = ['pg_only_ques_transformer', 'universal_transformer', 'transformer', 'seq2seq', 'copy_mechanism']
        fact_list = ['sr', 'ted', 'spg_transformer', 'pg_transformer', 'memnn']
        vspgt_list = ['mvspg_transformer', 'vspg_transformer']
        
        if model_type in no_fact_list:
            return ([_batch.src_ids, _batch.tar_ids], _batch.tar_loss_ids)
        elif model_type in fact_list:
            return ([_batch.src_ids, _batch.tar_ids, _batch.fact_ids], _batch.tar_loss_ids)
        elif model_type == 'multi_task':
            return ([_batch.src_ids, _batch.tar_ids, _batch.fact_tar_ids, _batch.fact_ids], \
                [_batch.tar_loss_ids, _batch.fact_tar_loss_ids, _batch.tar_loss_ids])
        elif model_type == 'multi_task_decex_transformer':
            return ([_batch.src_ids, _batch.tar_ids, _batch.fact_tar_ids, _batch.fact_ids], \
                [_batch.tar_loss_ids, _batch.fact_tar_loss_ids])
        elif model_type == 'merge_multi_task_decex_transformer':
            return ([_batch.src_ids, _batch.tar_ids, _batch.fact_tar_ids, _batch.fact_ids], \
                [_batch.tar_loss_ids])
        elif model_type in vspgt_list:
            if file_type == 'test':
                return ([_batch.src_ids, _batch.tar_ids, _batch.fact_ids, \
                    _batch.src_ids_exp, _batch.fact_ids_exp], _batch.tar_loss_ids_exp, _batch.oov_ids_dict)
            return ([_batch.src_ids, _batch.tar_ids, _batch.fact_ids, \
                _batch.src_ids_exp, _batch.fact_ids_exp], _batch.tar_loss_ids_exp)
        elif model_type == 'vspg_only_ques_transformer':
            if file_type == 'test':
                return ([_batch.src_ids, _batch.tar_ids, _batch.src_ids_exp], _batch.tar_loss_ids_exp, _batch.oov_ids_dict)
            return ([_batch.src_ids, _batch.tar_ids, _batch.src_ids_exp], _batch.tar_loss_ids_exp)

        elif model_type == 'cvspg_transformer':
            if file_type == 'test':
                return ([_batch.src_ids, _batch.tar_ids, _batch.conv_ids, _batch.fact_ids, \
                    _batch.src_ids_exp, _batch.fact_ids_exp], _batch.tar_loss_ids_exp, _batch.oov_ids_dict)
            return ([_batch.src_ids, _batch.tar_ids, _batch.conv_ids, _batch.fact_ids, \
                _batch.src_ids_exp, _batch.fact_ids_exp], _batch.tar_loss_ids_exp)
        elif model_type == 'fcvspg_transformer':
            if file_type == 'test':
                return ([_batch.src_ids, _batch.tar_ids, _batch.fact_ids, _batch.conv_ids, \
                    _batch.src_ids_exp, _batch.fact_ids_exp, _batch.conv_ids_exp], _batch.tar_loss_ids_exp, _batch.oov_ids_dict)
            return ([_batch.src_ids, _batch.tar_ids, _batch.fact_ids, _batch.conv_ids, \
                _batch.src_ids_exp, _batch.fact_ids_exp, _batch.conv_ids_exp], _batch.tar_loss_ids_exp)

        else:
            raise ValueError('The input model type: %s is not available. ' \
                'Please chech the file: data_reader.py line: _fit_model' % model_type)

    # This is a data generator, which is suitable for large-scale data set
    def data_generator(self, file_type, model_type):
        '''
        :param file_type: This is supposed to be: train, valid, or test
        :param model_type: This is supposed to be different models' name
        '''
        print('This is in data generator...')
        assert(self.max_src_len > 0)
        assert(self.max_tar_len > 0)
        assert(self.max_fact_len > 0)
        assert(self.max_conv_len > 0)
        assert file_type == 'train' or file_type == 'valid' or file_type == 'test'
    
        if file_type == 'train':
            qa_path = self.train_set_path
            conv_path = self.train_conv_path 
            fact_path = self.train_sent_fact_path
        elif file_type == 'valid':
            qa_path = self.valid_set_path
            conv_path = self.valid_conv_path 
            fact_path = self.valid_sent_fact_path
        elif file_type == 'test':
            qa_path = self.test_set_path
            conv_path = self.test_conv_path 
            fact_path = self.test_sent_fact_path

        batch = Batch()

        def _read_qa_fact(batch):
            while True:
                f_qa = open(qa_path)
                f_fact = open(fact_path)
                print(qa_path)
                print(fact_path)
                for index, (qa_line, fact_line) in enumerate(zip(f_qa, f_fact)):
                    qa_id = qa_line.strip().split('\t')[0]
                    fact_id = fact_line.strip().split('\t')[0]
                    assert (qa_id == fact_id)

                    # get oov dict first before deal with each line
                    oov_token2id, oov_id2token = self._get_oov_dict(qa_line, fact_line)

                    src, tar, tar_loss, src_exp, tar_exp, tar_loss_exp, src_tar, src_tar_loss = \
                        self._deal_qa_line(index, qa_line, oov_token2id)
                    no_fact_tag, facts, facts_exp, fact_tar, fact_tar_loss = self._deal_fact_line(index, fact_line, oov_token2id)

                    batch.src_ids.append(src)
                    batch.src_ids_exp.append(src_exp)
                    batch.tar_ids.append(tar)
                    batch.tar_ids_exp.append(tar_exp)
                    batch.tar_loss_ids.append(tar_loss)
                    batch.tar_loss_ids_exp.append(tar_loss_exp)
                    batch.fact_ids.append(facts)
                    batch.fact_ids_exp.append(facts_exp)

                    if no_fact_tag is True:
                        batch.fact_tar_ids.append(src_tar)
                        batch.fact_tar_loss_ids.append(src_tar_loss)
                    else:
                        batch.fact_tar_ids.append(fact_tar)
                        batch.fact_tar_loss_ids.append(fact_tar_loss)

                    batch.oov_ids_dict.append(oov_id2token)
    
                    if ((index + 1) % self.args.batch_size == 0):
                        #print('index + 1: ', index+1)
                        ret = deepcopy(batch)
                        #self._check_and_shuffle(ret)
                        ret.np_format()
                        batch = Batch()
                        yield self._fit_model(ret, file_type, model_type)

                if (len(batch.src_ids) != 0):
                    ret = deepcopy(batch)
                    #self._check_and_shuffle(ret)
                    ret.np_format()
                    batch = Batch()
                    yield self._fit_model(ret, file_type, model_type)

        def _read_qa_fact_conv(batch):
            while True:
                f_qa = open(qa_path)
                f_fact = open(fact_path)
                f_conv = open(conv_path)
                print(qa_path)
                print(fact_path)
                print(conv_path)
                for index, (qa_line, fact_line, conv_line) in enumerate(zip(f_qa, f_fact, f_conv)):
                    qa_id = qa_line.strip().split('\t')[0]
                    fact_id = fact_line.strip().split('\t')[0]
                    conv_id = conv_line.strip().split('\t')[0]
                    assert (qa_id == fact_id)
                    assert (qa_id == conv_id)

                    # get oov dict first before deal with each line
                    oov_token2id, oov_id2token = self._get_oov_dict(qa_line, fact_line)

                    src, tar, tar_loss, src_exp, tar_exp, tar_loss_exp, src_tar, src_tar_loss = \
                        self._deal_qa_line(index, qa_line, oov_token2id)
                    no_fact_tag, facts, facts_exp, fact_tar, fact_tar_loss = self._deal_fact_line(index, fact_line, oov_token2id)
                    no_conv_tag, convs, convs_exp, conv_tar, conv_tar_loss = self._deal_conv_line(index, conv_line, oov_token2id)

                    batch.src_ids.append(src)
                    batch.src_ids_exp.append(src_exp)
                    batch.tar_ids.append(tar)
                    batch.tar_ids_exp.append(tar_exp)
                    batch.tar_loss_ids.append(tar_loss)
                    batch.tar_loss_ids_exp.append(tar_loss_exp)
                    batch.fact_ids.append(facts)
                    batch.fact_ids_exp.append(facts_exp)
                    batch.conv_ids.append(convs)
                    batch.conv_ids_exp.append(convs_exp)

                    if no_fact_tag is True:
                        batch.fact_tar_ids.append(src_tar)
                        batch.fact_tar_loss_ids.append(src_tar_loss)
                    else:
                        batch.fact_tar_ids.append(fact_tar)
                        batch.fact_tar_loss_ids.append(fact_tar_loss)

                    batch.oov_ids_dict.append(oov_id2token)
    
                    if ((index + 1) % self.args.batch_size == 0):
                        #print('index + 1: ', index+1)
                        ret = deepcopy(batch)
                        #self._check_and_shuffle(ret)
                        ret.np_format()
                        batch = Batch()
                        yield self._fit_model(ret, file_type, model_type)

                if (len(batch.src_ids) != 0):
                    ret = deepcopy(batch)
                    #self._check_and_shuffle(ret)
                    ret.np_format()
                    batch = Batch()
                    yield self._fit_model(ret, file_type, model_type)

        if self.args.use_conv == True:
            return _read_qa_fact_conv(batch)
        else:
            return _read_qa_fact(batch)


    def __read_global_ids(self, token_path):
        f = open(token_path)
        token_ids = dict()
        id_tokens = dict()
        vocab_size = 0
        for line in f:
            elems = line.strip().split('\t')
            word = elems[0]
            index = int(elems[1])
            token_ids[word] = index
            id_tokens[index] = word 
            vocab_size += 1

        self.start_id = token_ids.get(config.START_TOKEN, -1)
        self.end_id = token_ids.get(config.END_TOKEN, -1)
        self.pad_id = token_ids.get(config.PAD_TOKEN, -1)
        self.unk_id = token_ids.get(config.UNK_TOKEN, -1)
        assert(self.start_id != -1)
        assert(self.end_id != -1)
        assert(self.pad_id != -1)
        assert(self.unk_id != -1)

        return token_ids, id_tokens, vocab_size


