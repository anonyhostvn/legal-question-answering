from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import T_co

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, TEST_IDX, SEGMENTED_TITLE_PATH
from sentence_transformers import InputExample, SentencesDataset
import numpy as np
import json
import random
import torch

from utilities.utils import transform_seg2uni


class SimSentDataset(Dataset):

    def __init__(self, lis_examples):
        self.lis_example = lis_examples

    def __len__(self):
        return len(self.lis_example)

    def __getitem__(self, index):
        return self.lis_example[index]


class DataPreprocess:
    def __init__(self, segmented_corpus_path, segmented_ques_path, test_idx_path, segmented_title_path):
        self.bm25_ranking = bm25_ranking
        with open(segmented_corpus_path, 'r') as f:
            self.segmented_corpus = json.load(f)
        with open(segmented_ques_path, 'r') as f:
            self.segmented_ques = json.load(f)
        with open(test_idx_path, 'r') as f:
            self.lis_test_idx = json.load(f)
        with open(segmented_title_path, 'r') as f:
            self.segmented_title = json.load(f)

    def generate_data_for_semantic_sim(self, rand_percent=0.5):
        lis_train_example = []
        for corpus_id in range(len(self.segmented_corpus)):
            title_txt = ' '.join([tok for sent in self.segmented_title[corpus_id] for tok in sent])
            n_sent = len(self.segmented_corpus[corpus_id])
            lis_rand_sent = np.random.randint(low=0, high=n_sent - 1, size=int(n_sent * rand_percent))
            for i_sent in lis_rand_sent:
                segmented_sent = self.segmented_corpus[corpus_id][i_sent]
                txt_sent = ' '.join(segmented_sent)
                lis_train_example.append(InputExample(texts=[title_txt, txt_sent], label=1.0))

            for _ in range(len(lis_rand_sent)):
                rand_corpus_id = random.randint(0, len(self.segmented_corpus) - 1)
                while rand_corpus_id != corpus_id and len(self.segmented_corpus[rand_corpus_id]) <= 1:
                    rand_corpus_id = random.randint(0, len(self.segmented_corpus) - 1)
                rand_sent_id = random.randint(0, len(self.segmented_corpus[rand_corpus_id]) - 1)
                segmented_sent = self.segmented_corpus[rand_corpus_id][rand_sent_id]
                txt_sent = ' '.join(segmented_sent)
                lis_train_example.append(InputExample(texts=[title_txt, txt_sent], label=0.0))
        return lis_train_example

    def generate_dataloader_for_semantic_sim(self):

        def custom_collate_fn(batch_data):
            return batch_data

        train_examples = self.generate_data_for_semantic_sim()
        sim_sent_dataset = SimSentDataset(lis_examples=train_examples)
        return DataLoader(sim_sent_dataset, shuffle=True, batch_size=16, collate_fn=custom_collate_fn)

    def test_bm25(self):
        print(transform_seg2uni([self.segmented_ques[0]]))
        lis_aid = self.bm25_ranking.get_ranking(query_idx=0, prefix='train_ques', top_n=20)
        lis_article = [self.segmented_corpus[aid] for aid in lis_aid]
        for article in transform_seg2uni(lis_article):
            print(article)
            print('-' * 20)


if __name__ == '__main__':
    data_preprocess = DataPreprocess(segmented_corpus_path=SEGMENTED_LEGAL_CORPUS,
                                     segmented_ques_path=SEGMENTED_DATA_QUESTION,
                                     test_idx_path=TEST_IDX,
                                     segmented_title_path=SEGMENTED_TITLE_PATH)
    data = data_preprocess.generate_dataloader_for_semantic_sim()
