# !/usr/bin/env python
import os, sys
from .tokenizers import *
from typing import Iterable, List, Optional

# generate checkpoint folder
def makedirs(fld):
    if not os.path.exists(fld):
        os.makedirs(fld)

def split_ex(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def get_stop_words(data_path):
    print('reading stop words...')
    dic_stop_words = dict()
    with open(data_path) as stopwords_file:
        for line in stopwords_file:
            dic_stop_words[line.strip()] = 1
    return dic_stop_words

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    
    return False

def clear_string(str_in): 
    return clean_str(str_in)

def pad_with_start_end(
               sample: Iterable[int],
               required_sequence_length: int,
               start_id: int,
               end_id: int,
               pad_id: int):
    
    # 2 means: <START> and <END>
    seq_length = required_sequence_length
    new_s = [start_id] + sample + [end_id]
    new_s += [pad_id] * (seq_length - len(new_s))
    return new_s

def pad_with_end(
               sample: Iterable[int],
               required_sequence_length: int,
               end_id: int,
               pad_id: int):
    
    seq_length = required_sequence_length
    new_s = sample + [end_id]
    new_s += [pad_id] * (seq_length - len(new_s))
    return new_s

def pad_with_start(
               sample: Iterable[int],
               required_sequence_length: int,
               start_id: int,
               pad_id: int):
    
    seq_length = required_sequence_length
    new_s = [start_id] + sample
    new_s += [pad_id] * (seq_length - len(new_s))
    return new_s

def pad_with_pad(
               sample: Iterable[int],
               required_sequence_length: int,
               pad_id: int):
    # keep the same length with the 'pad_with_start_end'
    seq_length = required_sequence_length
    new_s = sample + [pad_id] * (seq_length - len(sample))
    return new_s

class Hypothesis:
    def __init__(self, batch_size, tar_len, start_id, end_id):
        self.batch_size = batch_size
        self.tar_len = tar_len
        self.start_id = start_id
        self.end_id = end_id

        # bs, len
        self.res_ids = []
        self.pred_ids = []
        self.probs = []

        for i in range(batch_size):
            self.res_ids.append([])
            self.pred_ids.append([])
            self.probs.append([])

        for i in range(batch_size):
            self.res_ids[i].append(start_id)
            self.pred_ids[i].append(start_id)
            self.probs[i].append(0)

    def get_predictable_vars(self, pad_id):
        res = np.zeros((self.batch_size, self.tar_len))
        np_pred_ids = np.asarray(self.pred_ids)
        if len(np_pred_ids.shape) == 1:
            np_pred_ids = np.reshape(np_pred_ids, (np_pred_ids.shape[0], 1))
        for i in range(np_pred_ids.shape[1]):
            res[:, i] = np_pred_ids[:, i]
        for i in range(np_pred_ids.shape[1], self.tar_len):
            res[:, i] = pad_id
        return res

    @property
    def sum_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        sum_probs = []
        for bs_idx, bs_prob in enumerate(self.probs):
            sum_prob = sum(bs_prob)
            sum_probs.append(sum_prob)
        return sum_probs

    @property
    def length_norm(self):
        len_norms = []
        for res in self.res_ids:
            count = 0
            for _id in res:
                count += 1
                if _id == self.end_id:
                    break
            len_rate = count * 1.0 / self.tar_len
            len_norms.append(len_rate)
        return np.asarray(len_norms)

    @property
    def avg_prob(self):
        sum_probs = self.sum_prob
        sum_probs = np.asarray(sum_probs)
        sum_probs = sum_probs / len(self.res_ids[0])

        return sum_probs 


if __name__ == '__main__':
    s = '<p> each of rankins congressional terms coincided with initiation of u . s . military intervention in each of the       two world wars . a lifelong pacifist and a supporter of non-interventionism , [ 3 ] she was one of 50 house members , along with 6 senato      rs , who opposed the war declaration of 1917 , and the only member of congress to vote against declaring war on japan after the attack on       pearl harbor in 1941 . [ 4 ] [ 5 ]'
    print(clear_string(s))
