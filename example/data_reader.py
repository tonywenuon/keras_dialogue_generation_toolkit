
import sys, os
project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import argparse
import numpy as np
from sklearn.utils import shuffle
from typing import Callable, Optional, Sequence, Iterable
from run_script.args_parser import seq2seq_attn_add_arguments
from commonly_used_code import config, helper_fn

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
        elif self.args.data_set == 'reddit':
            self.train_set_path = config.reddit_train_qa_path
            self.train_conv_path = config.reddit_train_conv_path
            self.train_sent_fact_path = config.reddit_train_sent_fact_path
            self.valid_set_path = config.reddit_valid_qa_path
            self.valid_conv_path = config.reddit_valid_conv_path
            self.valid_sent_fact_path = config.reddit_valid_sent_fact_path
            self.test_set_path = config.reddit_test_qa_path
            self.test_conv_path = config.reddit_test_conv_path
            self.test_sent_fact_path = config.reddit_test_sent_fact_path
            self.src_global_token_path = config.reddit_global_token_path
            self.tar_global_token_path = config.reddit_global_token_path
        elif self.args.data_set == 'de_en':
            self.train_set_path = config.de_en_train_qa_path
            self.valid_set_path = config.de_en_valid_qa_path
            self.test_set_path = config.de_en_test_qa_path

            self.train_conv_path = None
            self.train_sent_fact_path = None
            self.valid_conv_path = None
            self.valid_sent_fact_path = None
            self.test_conv_path = None
            self.test_sent_fact_path = None

            self.src_global_token_path = config.de_en_src_global_token_path
            self.tar_global_token_path = config.de_en_tar_global_token_path

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

    # read all of the data to the memory. This is suitable for small data set
    def read_file(self, file_type, max_src_len, max_tar_len, 
                        max_fact_len=30, max_conv_len=30, 
                        get_fact=False, 
                        get_conv=False,
                        get_one_hot=False):
        '''
        :param file_type: This is supposed to be: train, valid, or test
        :param max_src_len: This is maximem source (question) length
        :param max_tar_len: This is maximem target (anwser) length
        :param max_fact_len: This is maximem fact (external knowledge) length, should be the same with source
        :param max_conv_len: This is maximem conversation (context) length
        :param get_fact: This is a boolean value to indicate whether read fact file
        :param get_conv: This is a boolean value to indicate whether read conv file
        '''

        assert(max_src_len > 0)
        assert(max_tar_len > 0)
        assert(max_fact_len > 0)
        assert(max_conv_len > 0)
        assert file_type == 'train' or file_type == 'valid' or file_type == 'test'
        print('current file type: %s' % file_type)

        src_len = max_src_len - config.src_reserved_pos
        tar_len = max_tar_len - config.tar_reserved_pos

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
        
        # read source and target
        print(qa_path)
        f = open(qa_path)
        indexes = []
        source_ids = []
        target_ids = []
        target_loss_ids = [] # Use to calculate loss. Only END sign, dont have START sign
        for line in f:
            elems = line.strip().split('\t')
            if len(elems) < 3:
                raise ValueError('Exceptd input to be 3 dimension, but received %d' % len(elems))

            indexes.append(int(elems[0].strip()))
            text = elems[1].strip()
            seq = [self.src_token_ids.get(token, self.unk_id) for token in text.split()]
            seq = seq[:src_len]
            new_seq = helper_fn.pad_with_start_end(seq, max_src_len, self.start_id, self.end_id, self.pad_id)
            source_ids.append(new_seq)
    
            text = elems[2].strip()
            seq = [self.tar_token_ids.get(token, self.unk_id) for token in text.split()]
            seq = seq[:tar_len]
            new_seq = helper_fn.pad_with_start(seq, max_tar_len, self.start_id, self.pad_id)
            target_ids.append(new_seq)
            new_seq = helper_fn.pad_with_end(seq, max_tar_len, self.end_id, self.pad_id)
            target_loss_ids.append(new_seq)
        f.close()
        if get_one_hot == True:
            target_one_hot = np.zeros((len(target_ids), len(target_ids[0]), self.vocab_size), dtype='int32')
            for i, target in enumerate(target_ids):
                for t, term_idx in enumerate(target):
                    if t > 0:
                        intaa = 0
                        target_one_hot[i, t - 1, term_idx] = 1
            target_loss_ids = target_one_hot

        pad_seqs = helper_fn.pad_with_pad([self.pad_id], max_fact_len, self.pad_id)
        facts_ids = []
        if get_fact == True:
            print(fact_path)
            with open(fact_path) as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    fact_ids = []
                    elems = line.split('\t')
                    # if there is no fact, add pad sequence
                    if elems[1] == config.NO_FACT:
                        fact_ids.append(pad_seqs)
                    else:
                        for text in elems[1:]:
                            seq = [self.src_token_ids.get(token, self.unk_id) for token in text.split()]
                            seq = seq[:max_fact_len]
                            new_seq = helper_fn.pad_with_pad(seq, max_fact_len, self.pad_id)
                            fact_ids.append(new_seq)
                    facts_ids.append(fact_ids)
            # keep facts to be the same number. If there is no so many fact, use pad_id as fact to pad it.
            facts_ids_tmp = []
            for facts in facts_ids:
                facts = facts[:self.args.fact_number]
                facts = facts + [pad_seqs] * (self.args.fact_number - len(facts))
                facts_ids_tmp.append(facts)
            facts_ids = facts_ids_tmp
    
        #pad_convs = [self.pad_id] * max_conv_len
        pad_seqs = helper_fn.pad_with_pad([self.pad_id], max_conv_len, self.pad_id)
        convs_ids = []
        if get_conv == True:
            print(conv_path)
            with open(conv_path) as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    conv_ids = []
                    elems = line.split('\t')
                    # if there is no context, add pad sequence
                    if elems[1] == config.NO_CONTEXT:
                        conv_ids.append(pad_seqs)
                    else:
                        for text in elems[1:]:
                            seq = [self.src_token_ids.get(token, self.unk_id) for token in text.split()]
                            seq = seq[:max_conv_len]
                            new_seq = helper_fn.pad_with_pad(seq, max_conv_len, self.pad_id)
                            conv_ids.append(new_seq)
                    convs_ids.append(conv_ids)
            # keep conv to be the same number. If there is no so many conv, use pad_id as conv to pad it.
            convs_ids_tmp = []
            for convs in convs_ids:
                convs = convs[:self.args.conv_number]
                convs = convs + [pad_seqs] * (self.args.conv_number- len(convs))
                convs_ids_tmp.append(convs)
            convs_ids = convs_ids_tmp
    
        assert(len(source_ids) == len(indexes))
        assert(len(source_ids) == len(target_ids))
        if get_fact == True:
            assert(len(source_ids) == len(facts_ids))
        if get_conv == True:
            assert(len(source_ids) == len(convs_ids))
    
        ## [[[ if for Zeyang to output ordered index, not shuffiling.
        #if get_fact == True and get_conv == True:
        #    indexes, source_ids, target_ids, convs_ids, facts_ids = shuffle(indexes, source_ids, target_ids, convs_ids, facts_ids)
        #elif get_fact == True:
        #    indexes, source_ids, target_ids, facts_ids = shuffle(indexes, source_ids, target_ids, facts_ids)
        #else:
        #    indexes, source_ids, target_ids = shuffle(indexes, source_ids, target_ids)
        ## ]]]
    
        return indexes, source_ids, target_ids, target_loss_ids, convs_ids, facts_ids

    # This is a data generator, which is suitable for large-scale data set
    def data_generator(self, file_type, model_type, max_src_len, max_tar_len, 
                        max_fact_len=30, max_conv_len=30, 
                        get_fact=False, 
                        get_conv=False
                        ):
        '''
        :param file_type: This is supposed to be: train, valid, or test
        :param max_src_len: This is maximem source (question) length
        :param max_tar_len: This is maximem target (anwser) length
        :param max_fact_len: This is maximem fact (external knowledge) length, which should be the same with source
        :param max_conv_len: This is maximem conversation (context) length
        :param get_fact: This is a boolean value to indicate whether read fact file
        :param get_conv: This is a boolean value to indicate whether read conv file
        '''
        print('This is in data generator...')
        assert(max_src_len > 0)
        assert(max_tar_len > 0)
        assert(max_fact_len > 0)
        assert(max_conv_len > 0)
        assert file_type == 'train' or file_type == 'valid' or file_type == 'test'
    
        src_len = max_src_len - config.src_reserved_pos
        tar_len = max_tar_len - config.tar_reserved_pos

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
       
        def _deal_qa(f):
            source_ids = []
            target_ids = []
            target_loss_ids = [] # Use to calculate loss. Only END sign, dont have START sign
            for index, line in enumerate(f):
                elems = line.strip().split('\t')
                text = elems[1].strip()
                seq = [self.src_token_ids.get(token, self.unk_id) for token in text.split()]
                #seq = [self.src_token_ids.get(token, self.pad_id) for token in text.split()]
                seq = seq[:src_len]
                new_seq = helper_fn.pad_with_start_end(seq, max_src_len, self.start_id, self.end_id, self.pad_id)
                source_ids.append(new_seq)
    
                text = elems[2].strip()
                seq = [self.tar_token_ids.get(token, self.unk_id) for token in text.split()]
                #seq = [self.tar_token_ids.get(token, self.pad_id) for token in text.split()]
                seq = seq[:tar_len]
                new_seq = helper_fn.pad_with_start(seq, max_tar_len, self.start_id, self.pad_id)
                target_ids.append(new_seq)
                new_seq = helper_fn.pad_with_end(seq, max_tar_len, self.end_id, self.pad_id)
                target_loss_ids.append(new_seq)

                if ((index + 1) % self.args.batch_size == 0):
                    res1 = np.asarray(source_ids)
                    res2 = np.asarray(target_ids) 
                    res3 = np.asarray(target_loss_ids)
                    res3 = np.reshape(res3, (res3.shape[0], res3.shape[1], 1))
                    source_ids = []
                    target_ids = []
                    target_loss_ids = [] 
                    yield res1, res2, res3
            if len(source_ids) != 0:
                res1 = np.asarray(source_ids)
                res2 = np.asarray(target_ids) 
                res3 = np.asarray(target_loss_ids)
                res3 = np.reshape(res3, (res3.shape[0], res3.shape[1], 1))
                source_ids = []
                target_ids = []
                target_loss_ids = []
                yield res1, res2, res3

        pad_fact_seqs = helper_fn.pad_with_pad([self.pad_id], max_fact_len, self.pad_id)
        def _deal_facts(f):
            facts_ids = []
            for index, line in enumerate(f):
                line = line.strip()
                fact_ids = []
                elems = line.split('\t')
                # if there is no fact, add pad sequence
                if elems[1] == config.NO_FACT:
                    fact_ids.append(pad_fact_seqs)
                else:
                    for text in elems[1:]:
                        seq = [self.src_token_ids.get(token, self.unk_id) for token in text.split()]
                        #seq = [self.src_token_ids.get(token, self.pad_id) for token in text.split()]
                        seq = seq[:max_fact_len]
                        new_seq = helper_fn.pad_with_pad(seq, max_fact_len, self.pad_id)
                        fact_ids.append(new_seq)
                # pad fact number
                fact_ids = fact_ids[:self.args.fact_number]
                fact_ids = fact_ids + [pad_fact_seqs] * (self.args.fact_number - len(fact_ids))
                facts_ids.append(fact_ids)
                if ((index + 1) % self.args.batch_size == 0):
                    res = np.asarray(facts_ids)
                    facts_ids = []
                    yield res
            if len(facts_ids) != 0:
                res = np.asarray(facts_ids)
                facts_ids = []
                yield res

        pad_conv_seqs = helper_fn.pad_with_pad([self.pad_id], max_conv_len, self.pad_id)
        def _deal_convs(f):
            convs_ids = []
            for index, line in enumerate(f):
                line = line.strip()
                conv_ids = []
                elems = line.split('\t')
                # if there is no context, add pad sequence
                if elems[1] == config.NO_CONTEXT:
                    conv_ids.append(pad_conv_seqs)
                else:
                    for text in elems[1:]:
                        seq = [self.src_token_ids.get(token, self.unk_id) for token in text.split()]
                        #seq = [self.src_token_ids.get(token, self.pad_id) for token in text.split()]
                        seq = seq[:max_conv_len]
                        new_seq = helper_fn.pad_with_pad(seq, max_conv_len, self.pad_id)
                        conv_ids.append(new_seq)
                conv_ids = conv_ids[:self.args.conv_number]
                conv_ids = conv_ids + [pad_conv_seqs] * (self.args.conv_number- len(conv_ids))
                convs_ids.append(conv_ids)
                if ((index + 1) % self.args.batch_size == 0):
                    res = np.asarray(convs_ids)
                    convs_ids = []
                    yield res
            if len(convs_ids) != 0:
                res = np.asarray(convs_ids)
                convs_ids = []
                yield res

        def _check_and_shuffle(source_ids, target_ids, target_loss_ids, convs_ids, facts_ids):
            assert(len(source_ids) == len(target_ids))
            assert(len(source_ids) == len(target_loss_ids))
            if get_fact == True:
                assert(len(source_ids) == len(facts_ids))
            if get_conv == True:
                assert(len(source_ids) == len(convs_ids))
    
            if get_fact == True and get_conv == True:
                source_ids, target_ids, target_loss_ids, convs_ids, facts_ids = \
                    shuffle(source_ids, target_ids, target_loss_ids, convs_ids, facts_ids)
            elif get_fact == True:
                source_ids, target_ids, target_loss_ids, facts_ids = shuffle(source_ids, target_ids, target_loss_ids, facts_ids)
            else:
                source_ids, target_ids, target_loss_ids = shuffle(source_ids, target_ids, target_loss_ids)

            return (source_ids, target_ids, target_loss_ids, convs_ids, facts_ids)

        def _fit_model(res_src, res_tar, res_tar_loss, res_conv, res_fact):
            '''
            Please carefully choose the output type to fit with your model's inputs
            '''
            no_fact_list = ['pg_only_ques_transformer', 'universal_transformer', 'transformer', 'seq2seq', 'copy_mechanism']
            fact_list = ['decex_transformer', 'vspg_transformer', 'spg_transformer', 'pg_transformer', 'memnn']
            conv_fact_list = ['transformer_conv_fact', 'universal_transformer_conv_fact']
            if model_type in no_fact_list:
                # only return question and answer as inputs
                return ([res_src, res_tar], res_tar_loss)
            elif model_type in fact_list:
                # return question, answer and facts as inputs
                return ([res_src, res_tar, res_fact], res_tar_loss)
            elif model_type in conv_fact_list:
                # return question, answer, context and facts as inputs
                return ([res_src, res_tar, res_conv, res_fact], res_tar_loss)
            else:
                raise ValueError('The input model type: %s is not available. ' \
                    'Please chech the file: data_reader.py line: _fit_model' % model_type)

        while True:
            source_ids, target_ids, target_loss_ids, convs_ids, facts_ids = None, None, None, None, None
            print(qa_path)
            f_qa = open(qa_path)

            res_src, res_tar, res_tar_loss, res_fact, res_conv = None, None, None, None, None
            if get_fact == True and get_conv == True:
                f_fact = open(fact_path)
                f_conv = open(conv_path)
                for ((source_ids, target_ids, target_loss_ids), facts_ids, convs_ids) in \
                    zip(_deal_qa(f_qa), _deal_facts(f_fact), _deal_convs(f_conv)):
                    res_src, res_tar, res_tar_loss, res_conv, res_fact = \
                        _check_and_shuffle(source_ids, target_ids, target_loss_ids, convs_ids, facts_ids)
                    yield _fit_model(res_src, res_tar, res_tar_loss, res_conv, res_fact) 
            elif get_fact == True:
                f_fact = open(fact_path)
                for ((source_ids, target_ids, target_loss_ids), facts_ids) in \
                    zip(_deal_qa(f_qa), _deal_facts(f_fact)):
                    res_src, res_tar, res_tar_loss, res_conv, res_fact = \
                        _check_and_shuffle(source_ids, target_ids, target_loss_ids, convs_ids, facts_ids)
                    yield _fit_model(res_src, res_tar, res_tar_loss, res_conv, res_fact) 
            else:
                for (source_ids, target_ids, target_loss_ids) in _deal_qa(f_qa):
                    res_src, res_tar, res_tar_loss, res_conv, res_fact = \
                        _check_and_shuffle(source_ids, target_ids, target_loss_ids, convs_ids, facts_ids)
                    yield _fit_model(res_src, res_tar, res_tar_loss, res_conv, res_fact) 

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    seq2seq_attn_add_arguments(parser)
    args = parser.parse_args()

    ds = DataSet(args)
    #ds.read_file('test', 
    #             max_src_len=30, 
    #             max_tar_len=30, 
    #             max_fact_len=30, 
    #             max_conv_len=30, 
    #             get_fact=True, 
    #             get_conv=False,
    #             get_one_hot=True)

    # no_fact_list
    for index, ([source_ids, target_ids, facts_ids], target_loss_ids) in enumerate(ds.data_generator(
    # fact_list
    #for index, ([source_ids, target_ids, facts_ids], target_loss_ids) in enumerate(ds.data_generator(
    # conv fact list
    #for index, ([source_ids, target_ids, convs_ids, facts_ids], target_loss_ids) in enumerate(ds.data_generator(
                 'test', 'decex_transformer',
                 max_src_len=30, 
                 max_tar_len=30, 
                 max_fact_len=30, 
                 get_fact=True, 
                 )):
        print('*' * 100)
        print(index)
        #print(len(source_ids))
        #print(len(target_ids))
        #print(len(target_loss_ids))
        idx = 0
        print('source: ', source_ids[idx])
        print('target: ', target_ids[idx])
        print('target loss: ',target_loss_ids[idx])
        print('facts: ', facts_ids[idx])
        #print(len(facts_ids))
        #print(len(facts_ids[2]))
        #print(len(facts_ids[5]))
        #print(len(convs_ids))
        #print(len(convs_ids[4]))
        #print(len(convs_ids[5]))

