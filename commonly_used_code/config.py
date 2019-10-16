# !/use/bin/env python

import os
import configparser
from .helper_fn import *

parser = configparser.SafeConfigParser()
config_file_path = '../configuration/config.ini'
parser.read(config_file_path)

# reserve <START> and <END> for source
src_reserved_pos = 2
# reserve <START> or <END> for target
tar_reserved_pos = 1
NO_FACT = 'no_fact'
NO_CONTEXT = 'no_context'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SPECIAL_TOKENS = [START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN]

# original data set path
example_data_data_path = parser.get('FilePath', 'example_data_path')

train_path = parser.get('FilePath', 'symblic_single_turn_train')
valid_path = parser.get('FilePath', 'symblic_single_turn_valid')
test_path = parser.get('FilePath', 'symblic_single_turn_test')

# example_data data set
stop_words_path = parser.get('example_data', 'stop_words')
train_data_path = parser.get('example_data', 'train_data')
valid_data_path = parser.get('example_data', 'valid_data')
test_data_path = parser.get('example_data', 'test_data')

# generate symblic question answer and facts
use_for_retrieval_token_path = parser.get('SymblicQAF', 'retrieval_token_dict')
src_global_token_path = parser.get('SymblicQAF', 'src_global_token_dict')
tar_global_token_path = parser.get('SymblicQAF', 'tar_global_token_dict')
pro_qa_data_path = parser.get('SymblicQAF', 'pro_qa_data')
pro_facts_data_path = parser.get('SymblicQAF', 'pro_facts_data')
pro_conv_data_path = parser.get('SymblicQAF', 'pro_conv_data')
sent_fact_data_path = parser.get('SymblicQAF', 'sent_fact_data')

# IDF data path
facts_idf_path = parser.get('TFIDF', 'facts_idf_data')

# example_data train valid test data path
example_data_train_path = os.path.join(example_data_data_path, train_path)
example_data_valid_path = os.path.join(example_data_data_path, valid_path)
example_data_test_path = os.path.join(example_data_data_path, test_path)
makedirs(example_data_train_path)
makedirs(example_data_valid_path)
makedirs(example_data_test_path)

# original data of example_data
example_data_stop_words_path = os.path.join(example_data_data_path, stop_words_path)
example_data_train_data_path = os.path.join(example_data_data_path, train_data_path)
example_data_valid_data_path = os.path.join(example_data_data_path, valid_data_path)
example_data_test_data_path = os.path.join(example_data_data_path, test_data_path)

# used for example_data file path
example_data_use_for_retrieval_token_path = os.path.join(example_data_train_path, use_for_retrieval_token_path)
example_data_global_token_path = os.path.join(example_data_train_path, src_global_token_path)
example_data_facts_idf_path = os.path.join(example_data_train_path, facts_idf_path)

example_data_train_qa_path = os.path.join(example_data_train_path, pro_qa_data_path)
example_data_train_facts_path = os.path.join(example_data_train_path, pro_facts_data_path)
example_data_train_conv_path = os.path.join(example_data_train_path, pro_conv_data_path)
example_data_train_sent_fact_path = os.path.join(example_data_train_path, sent_fact_data_path)

example_data_valid_qa_path = os.path.join(example_data_valid_path, pro_qa_data_path)
example_data_valid_facts_path = os.path.join(example_data_valid_path, pro_facts_data_path)
example_data_valid_conv_path = os.path.join(example_data_valid_path, pro_conv_data_path)
example_data_valid_sent_fact_path = os.path.join(example_data_valid_path, sent_fact_data_path)

example_data_test_qa_path = os.path.join(example_data_test_path, pro_qa_data_path)
example_data_test_facts_path = os.path.join(example_data_test_path, pro_facts_data_path)
example_data_test_conv_path = os.path.join(example_data_test_path, pro_conv_data_path)
example_data_test_sent_fact_path = os.path.join(example_data_test_path, sent_fact_data_path)

