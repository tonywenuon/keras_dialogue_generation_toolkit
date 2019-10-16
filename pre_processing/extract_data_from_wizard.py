
import sys, queue
import json

sys.path.append('../')
from commonly_used_code import config
from commonly_used_code import helper_fn

class ExtractData:
    def __init__(self, 
                 json_path, 
                 qa_data_path, 
                 conv_data_path, 
                 facts_data_path, 
                 sent_fact_path, 
                 para_fact_path,
                 global_token_data_path=None 
                 ):

        print('current file:', qa_data_path)
        self.MAX_VOCAB_SIZE = 50000
        self.RESPONSE_MIN_LENGTH = 5
        # input data
        self.wizard_data_json = self.__load_json(json_path)

        # generate data
        self.global_token_data_path = global_token_data_path
        self.qa_data_path = qa_data_path
        self.conv_data_path = conv_data_path
        self.facts_data_path = facts_data_path
        self.sent_fact_path = sent_fact_path
        self.para_fact_path = para_fact_path
        
    def __load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def __generate_global_token(self):
        print('generate global token...')
        outobj = open(self.global_token_data_path, 'w')
        dic_tokens = {}
        dic_tokens_tmp = {}
        index = 0
        for token in config.SPECIAL_TOKENS:
            dic_tokens[token] = index
            index += 1
        print('read ori qa file...')
        with open(self.qa_data_path) as f:
            for line in f:
                elems = line.strip().split('\t')
                for i in range(1, len(elems)):
                    cur_sent = elems[i]
                    tokens = cur_sent.split(' ')
                    for token in tokens:
                        token = token.strip()
                        if token == '':
                            continue
                        dic_tokens_tmp[token] = dic_tokens_tmp.get(token, 0) + 1
        print('read ori fact file...')
        with open(self.facts_data_path) as f:
            for line in f:
                elems = line.strip().split('\t')
                for i in range(1, len(elems)):
                    cur_sent = elems[i]
                    tokens = cur_sent.split(' ')
                    for token in tokens:
                        token = token.strip()
                        if token == '':
                            continue
                        dic_tokens_tmp[token] = dic_tokens_tmp.get(token, 0) + 1

        pq = queue.PriorityQueue()
        for token in dic_tokens_tmp:
            pq.put((-dic_tokens_tmp[token], token))
        while not pq.empty():
            freq, token = pq.get()
            dic_tokens[token] = index
            index += 1
            if len(dic_tokens) == self.MAX_VOCAB_SIZE:
                break

        print('generate global token file...')
        for key, index in dic_tokens.items():
            write_line = '\t'.join([key, str(index)])
            write_line += '\n'
            outobj.write(write_line)

        outobj.close()

    def __check_whether_kept(self, res):
        if len(res.strip().split(' ')) < self.RESPONSE_MIN_LENGTH:
            return False
        return True

    def extract_data(self):
        print('-------------in extract data func--------------')
        qa_outobj = open(self.qa_data_path, 'w')
        conv_outobj = open(self.conv_data_path, 'w')
        facts_outobj = open(self.facts_data_path, 'w')
        sent_fact_outobj = open(self.sent_fact_path, 'w')
        para_fact_outobj = open(self.para_fact_path, 'w')

        tag_app = '_Apprentice'
        tag_wiz = '_Wizard'
        count = 0
        # enumerate each dialog
        for dialogs in self.wizard_data_json:
            cur_dialog = dialogs['dialog']
            # index: 0:topic dict; 1: partner dict; 2: self dict
            contexts = []
            for i in range(3):
                contexts.append([])
            texts = []
            facts = []
            sents = []
            paras = []
            no_fact = "no_passages_used"
            for index, dialog in enumerate(cur_dialog):
                # string
                speaker = dialog['speaker']
                is_wizard = speaker.find(tag_wiz)
                is_apprentice = speaker.find(tag_app)
                # string
                text = dialog['text']
                if is_wizard != -1:
                    # dict
                    checked_sentence = dialog['checked_sentence']
                    # dict. the value relates to the key of retrieved_passages
                    checked_passage = dialog['checked_passage']
                # list. 7 articles. each article is a dict which the key is title, values is sentence list
                retrieved_passages = dialog['retrieved_passages']
                # list, 7 articles' title
                retrieved_topics = dialog['retrieved_topics']

                # update the retrieved source
                if index == 0:
                    contexts[0] = retrieved_passages
                else:
                    if is_wizard != -1:
                        contexts[2] = retrieved_passages
                    if is_apprentice != -1:
                        contexts[1] = retrieved_passages

                if index == 0 and is_wizard != -1:
                    # the first one and it is wizard, then take knowledge as question
                    for key, value in checked_sentence.items():
                        texts.append(value)
                        facts.append(no_fact)
                        sents.append(no_fact)
                        paras.append(no_fact)
                texts.append(text)
                if is_wizard != -1:
                    # get the chosen sentence
                    chosen_sent = no_fact
                    # only one sentence here actually
                    for key, value in checked_sentence.items():
                        chosen_sent = value
                    sents.append(chosen_sent)

                    # get the chosen passage
                    passage_key = ''
                    passage_value = ''
                    # only one value actually
                    for key, value in checked_passage.items():
                        passage_key = key
                        passage_value = value
                    search_passages = contexts[0]
                    if passage_key.startswith('partner'):
                        search_passages = contexts[1]
                    elif passage_key.startswith('self'):
                        search_passages = contexts[2]
                    chosen_para = no_fact
                    for dic_article in search_passages:
                        if passage_value in dic_article.keys():
                            chosen_para = ' '.join(dic_article[passage_value])
                            break
                    paras.append(chosen_para)
                    
                    # get all of the candidates passages
                    candidates = []
                    for i in range(2, 0, -1):
                        context = contexts[i]
                    #for context in contexts:
                        for dic_article in context:
                            para_fact = []
                            for _, item in dic_article.items():
                                para_fact.append(' '.join(item))
                            if len(para_fact) > 0:
                                para_fact_line = '\t'.join(para_fact)
                                candidates.append(para_fact_line)
                    all_facts = '\t'.join(candidates)    
                    facts.append(all_facts)

                else:
                    facts.append(no_fact)
                    sents.append(no_fact)
                    paras.append(no_fact)

            assert(len(texts) == len(facts))
            assert(len(texts) == len(sents))
            assert(len(texts) == len(paras))

            conv = ''
            pre_qa = ''
            for i in range(0, len(texts) - 1, 2):
                # clean the question and answer
                ques = texts[i].strip()
                res = texts[i + 1].strip()
                tmp = ques.split('\t')
                ques = ' '.join(tmp)
                tmp = res.split('\t')
                res = ' '.join(tmp)
                ques = helper_fn.clear_string(ques)
                res = helper_fn.clear_string(res)

                qa = '\t'.join([str(count), ques, res])
                write_line = qa.strip() + '\n'
                write_line = write_line.lower()
                qa_outobj.write(write_line)

                tmp = []
                if i == 0:
                    #conv = ques
                    conv = config.NO_CONTEXT
                    tmp.append(conv)
                else:
                    elems = conv.strip().split('\t')
                    for sent in elems[1:]:
                        sent = helper_fn.clear_string(sent)
                        tmp.append(sent)
                join_str = '\t'.join(tmp)
                conv = '\t'.join([pre_qa.strip(), join_str.strip()])
                conv = '\t'.join([str(count), conv.strip()])
                write_line = conv.strip() + '\n'
                write_line = write_line.lower()
                conv_outobj.write(write_line)

                elems = facts[i + 1].strip().split('\t')
                tmp = []
                for sent in elems:
                    sent = helper_fn.clear_string(sent)
                    tmp.append(sent)
                join_str = '\t'.join(tmp)
                fact = '\t'.join([str(count), join_str])
                write_line = fact.strip() + '\n'
                write_line = write_line.lower()
                facts_outobj.write(write_line)

                elems = sents[i + 1].strip().split('\t')
                tmp = []
                for sent in elems:
                    sent = helper_fn.clear_string(sent)
                    tmp.append(sent)
                join_str = '\t'.join(tmp)
                sent = '\t'.join([str(count), join_str])
                write_line = sent.strip() + '\n'
                write_line = write_line.lower()
                sent_fact_outobj.write(write_line)

                elems = paras[i + 1].strip().split('\t')
                tmp = []
                for sent in elems:
                    sent = helper_fn.clear_string(sent)
                    tmp.append(sent)
                join_str = '\t'.join(tmp)
                para = '\t'.join([str(count), join_str])
                write_line = para.strip() + '\n'
                write_line = write_line.lower()
                para_fact_outobj.write(write_line)

                count += 1
                # keep the last utterance ahead of the previous utterances
                pre_qa = '\t'.join([res, ques])

        qa_outobj.close()        
        conv_outobj.close()        
        facts_outobj.close()
        sent_fact_outobj.close()        
        para_fact_outobj.close()        

        if self.global_token_data_path != None:
            self.__generate_global_token()

if __name__ == '__main__':
    print('Extracting data from Wizard of Wikipedia...')
    ed = ExtractData(config.wizard_train_data_path, 
                     config.wizard_train_qa_path, 
                     config.wizard_train_conv_path, 
                     config.wizard_train_facts_path, 
                     config.wizard_train_ground_truth_sent_fact_path, 
                     config.wizard_train_ground_truth_para_fact_path,
                     config.wizard_global_token_path 
                     )
    ed.extract_data()

    ed = ExtractData(config.wizard_valid_data_path, 
                     config.wizard_valid_qa_path, 
                     config.wizard_valid_conv_path, 
                     config.wizard_valid_facts_path, 
                     config.wizard_valid_ground_truth_sent_fact_path, 
                     config.wizard_valid_ground_truth_para_fact_path
                     )
    ed.extract_data()

    ed = ExtractData(config.wizard_test_data_path, 
                     config.wizard_test_qa_path, 
                     config.wizard_test_conv_path, 
                     config.wizard_test_facts_path, 
                     config.wizard_test_ground_truth_sent_fact_path, 
                     config.wizard_test_ground_truth_para_fact_path
                     )
    ed.extract_data()

