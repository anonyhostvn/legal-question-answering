from tqdm import tqdm
import os
from random import shuffle
import json

from finetunning_bert.const import CORPUS_PATH, BERT_CORPUS_PATH, TRAIN_IDX_PATH, TEST_IDX_PATH


class DataPreprocess:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.output_file = None

    def start_make_bert_corpus(self, corpus_path, output_path):
        with open(corpus_path, 'r') as f:
            segmented_corpus = json.load(f)

        with open(output_path, 'w') as bert_corpus_file:
            self.output_file = bert_corpus_file
            for article in tqdm(segmented_corpus):
                if article is not None:
                    self.process_article(article)

    def write_text_output(self, txt):
        assert self.output_file is not None
        assert len(txt.split(' ')) <= self.max_seq_length + 5, f'Length of txt is {len(txt.split(" "))}'
        self.output_file.write(txt + '\n')

    def sent_split_out(self, sent):
        for sid in range(0, len(sent), self.max_seq_length):
            recent_text = sent[sid:min(sid + self.max_seq_length, len(sent))]
            self.write_text_output(' '.join(recent_text))

    def process_article(self, article):
        recent_text = ''
        recent_length = 0
        for sent in article:
            #  Trường hợp độ dài của câu bé hơn MAX_SEQ_LENGTH
            if len(sent) <= self.max_seq_length:
                # Trường khi thêm câu vào câu hiện có thì độ dài vẫn bé hơn MAX_SEQ_LENGTH
                # ==> Thêm câu vào chuỗi hiện tại
                if recent_length + len(sent) <= self.max_seq_length:
                    recent_length += len(sent)
                    recent_text += ' '.join(sent)
                # Trường hượp khi thêm câu vào câu hiện có thì độ dài lớn hơn MAX_SEQ_LENGTH
                # ==> Thêm câu hiện tại vào corpus và tạo 1 câu hiện tại mới
                else:
                    # print(recent_text)
                    self.write_text_output(recent_text)
                    recent_text = ' '.join(sent)
                    recent_length = len(sent)
            # Trường hợp độ dài của câu đó đã lớn hơn MAX_SEQ_LENGTH
            # ==> Ghi câu hiện tại vào corpus + thực hiện việc chia câu
            else:
                # Nêu câu hiện tại vẫn đang xây dựng dở
                # ==> Thêm câu hiện tại vào corpus
                if recent_length > 0:
                    self.write_text_output(recent_text)
                    recent_text = ''
                    recent_length = 0
                self.sent_split_out(sent)

        if recent_length > 0:
            self.write_text_output(recent_text)

    def test_preprocess(self, bert_corpus_path):
        is_all_ok = True
        with open(bert_corpus_path, 'r') as f:
            all_line = f.readlines()
            print('Tổng số sample là ', len(all_line))
            for line in all_line:
                if len(line.split(' ')) > self.max_seq_length:
                    print(len(line.split(' ')))
                    is_all_ok = False

        if is_all_ok:
            print('OK')


def train_test_split_idx(corpus_path, train_idx_path, test_idx_path, test_size=0.2):
    assert os.path.exists(corpus_path)
    assert test_size < 1
    with open(corpus_path, 'r') as corpus_file:
        samples = corpus_file.readlines()
    list_idx = list(range(len(samples)))
    shuffle(list_idx)
    split_point = int(len(list_idx) * test_size)

    lis_test_idx = list_idx[:split_point]
    with open(test_idx_path, 'w') as test_idx_file:
        json.dump(lis_test_idx, test_idx_file)

    lis_train_idx = list_idx[split_point:]
    with open(train_idx_path, 'w') as train_idx_file:
        json.dump(lis_train_idx, train_idx_file)


if __name__ == '__main__':
    MAX_PHOBERT_SEQ_LENGTH = 256
    # data_preprocess = DataPreprocess(max_seq_length=MAX_PHOBERT_SEQ_LENGTH)
    # data_preprocess.start_make_bert_corpus(corpus_path=CORPUS_PATH, output_path=BERT_CORPUS_PATH)
    # data_preprocess.test_preprocess(bert_corpus_path=BERT_CORPUS_PATH)
    train_test_split_idx(corpus_path=BERT_CORPUS_PATH, train_idx_path=TRAIN_IDX_PATH, test_idx_path=TEST_IDX_PATH)
    pass
