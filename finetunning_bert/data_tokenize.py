from transformers import AutoTokenizer

from finetunning_bert.const import BERT_CORPUS_PATH, TOKENIZED_OUTPUT_PATH
import pickle

TOKENIZER = 'vinai/phobert-base'


class DataTokenize:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def start_tokenize(self, bert_corpus_path, tokenized_output_path):
        with open(bert_corpus_path, 'r') as bert_corpus_file:
            all_sample = bert_corpus_file.readlines()
            all_sample = all_sample[:10]

        tokenized_output = self.tokenizer(all_sample, padding='max_length', truncation='only_first')
        with open(tokenized_output_path, 'wb') as tokenized_output_file:
            pickle.dump(tokenized_output, tokenized_output_file)


if __name__ == '__main__':
    data_tokenize = DataTokenize(TOKENIZER)
    data_tokenize.start_tokenize(bert_corpus_path=BERT_CORPUS_PATH, tokenized_output_path=TOKENIZED_OUTPUT_PATH)
    with open(TOKENIZED_OUTPUT_PATH, 'rb') as f:
        a = pickle.load(f)
