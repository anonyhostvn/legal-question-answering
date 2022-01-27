from global_config import STOP_WORD_PATH
from string import punctuation


class TextPreprocessing:
    def __init__(self):
        with open(STOP_WORD_PATH, 'r') as f:
            self.lis_stop_word = f.readlines()
            self.lis_stop_word = ['_'.join(line.strip().split(' ')) for line in self.lis_stop_word]

    def remove_stop_word(self, sent):
        return ' '.join([word for word in sent.split(' ') if word not in self.lis_stop_word])

    def remove_special_character(self, sent):
        cat_sent = ''.join([c for c in sent if c == ' ' or c == '_' or c not in punctuation])
        return ' '.join([word for word in cat_sent.strip().split(' ') if word != ''])

    def process(self, sent):
        s = self.remove_special_character(sent)
        s = self.remove_stop_word(s)
        return s


mono_text_preprocessing = TextPreprocessing()

if __name__ == '__main__':
    text_processing = TextPreprocessing()
    processed_text = text_processing.process(sent='Xin_chào, a_lô tôi là Long.')
    print(processed_text)
