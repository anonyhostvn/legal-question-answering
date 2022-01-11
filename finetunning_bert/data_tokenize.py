from transformers import AutoTokenizer

TOKENIZER = 'vinai/phobert-base'


class DataTokenize:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def start_tokenize(self, all_sample):
        return self.tokenizer(all_sample, padding='max_length', truncation='only_first',
                              return_tensors='pt')


if __name__ == '__main__':
    data_tokenize = DataTokenize(TOKENIZER)
