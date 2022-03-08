import os

STOP_WORD_PATH = os.path.join('large_files', 'vietnamese-stopwords.txt')
LEGAL_CORPUS_PATH = os.path.join('zac2021-ltr-data', 'legal_corpus.json')
DATA_QUESTION_PATH = os.path.join('zac2021-ltr-data', 'train_question_answer.json')
TEST_QUESTION_PATH = os.path.join('zac2021-ltr-data', 'public_test_question.json')
PRIVATE_TEST_QUESTION_PATH = os.path.join('zac2021-ltr-data', 'private_test_question.json')
SEGMENTED_LEGAL_CORPUS = os.path.join('segemented_data', 'segmented_corpus.json')
SEGMENTED_DATA_QUESTION = os.path.join('segemented_data', 'train_ques_segmented.json')
SEGMENTED_TEST_QUESTION = os.path.join('segemented_data', 'test_ques_segmented.json')
SEGMENTED_PRIVATE_TEST = os.path.join('segemented_data', 'private_ques_segmented.json')
SEGMENTED_TITLE_PATH = os.path.join('segemented_data', 'segmented_title.json')
TRAIN_IDX = os.path.join('data_spliter', 'train_idx.json')
TEST_IDX = os.path.join('data_spliter', 'test_idx.json')
RAW_LEGAL_TEXT_CORPUS_PATH = os.path.join('large_files', 'bert_corpus_path_v1.txt')
# PRETRAIN_BERT_NAME = 'vinai/phobert-base'
PRETRAIN_BERT_NAME = 'haisongzhang/roberta-tiny-cased'
# PRETRAIN_BERT_TOKENIZER = 'vinai/phobert-base'
PRETRAIN_BERT_TOKENIZER = 'haisongzhang/roberta-tiny-cased'
LEGAL_BERT_MLM = 'legal_bert_checkpoint'
SENT_BERT_CHECKPOINT = 'sent_bert_checkpoint'
SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH = 'sent_bert_downstream_checkpoint'
