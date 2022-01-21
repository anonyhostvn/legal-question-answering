import os

from rank_bm25 import BM25Okapi
import json

from global_config import SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, SEGMENTED_TEST_QUESTION
import numpy as np
from tqdm import tqdm
import pickle


class Bm25PreRanking:
    def __init__(self, cached_folder_path):
        self.bm25 = None
        self.cluster_segmented_query = {
            'train_ques': SEGMENTED_DATA_QUESTION,
            'test_ques': SEGMENTED_TEST_QUESTION
        }
        self.loaded_score = None
        self.loaded_prefix = None
        self.get_cached_path = lambda prefix: os.path.join(cached_folder_path,
                                                           f'{prefix}-pre_computed_score.npy')
        self.bm25_cached_path = os.path.join(cached_folder_path, 'bm25_cached.pkl')
        self.load_bm25()

    def get_top_n(self, tokenized_query, n):
        lis_score = self.bm25.get_scores(tokenized_query)
        top_n_idx = np.argsort(lis_score)[:n]
        return top_n_idx

    def build_and_save_bm25(self):
        with open(SEGMENTED_LEGAL_CORPUS, 'r') as f:
            segmented_legal_corpus = json.load(f)

        modified_segmented_legal_corpus = [[tok for sent in segmented_article for tok in sent]
                                           for segmented_article in tqdm(segmented_legal_corpus)]
        print('Computing BM25dict')
        bm25_okapi = BM25Okapi(modified_segmented_legal_corpus)

        print('Compute done ! save bm25 object by pickle')

        with open(self.bm25_cached_path, 'wb') as bm25_cached_file:
            pickle.dump(bm25_okapi, bm25_cached_file)

    def load_bm25(self):
        if not os.path.exists(self.bm25_cached_path):
            print('Bm25 cached is not exist, start process to compute bm25 object')
            self.build_and_save_bm25()
        with open(self.bm25_cached_path, 'rb') as bm25_cached_file:
            self.bm25 = pickle.load(bm25_cached_file)

    def pre_score_calculated(self, prefix):
        assert prefix in self.cluster_segmented_query.keys(), 'Prefix is not in segmented database'
        print('Start compute ranking score')

        with open(self.cluster_segmented_query.get(prefix), 'r') as f:
            lis_segmented_query = json.load(f)

        lis_score = []
        for segmented_query in tqdm(lis_segmented_query):
            modified_segmented_query = [tok for sent in segmented_query for tok in sent]
            lis_score.append(self.bm25.get_scores(modified_segmented_query))

        lis_score = np.array(lis_score)
        np.save(file=self.get_cached_path(prefix), arr=lis_score)

    def get_ranking(self, query_idx: int, prefix: str, top_n: int):
        if not os.path.exists(self.get_cached_path(prefix)):
            self.pre_score_calculated(prefix)

        if self.loaded_prefix != prefix:
            self.loaded_prefix = prefix
            self.loaded_score = np.load(self.get_cached_path(prefix))

        query_lis_score = self.loaded_score[query_idx]
        lis_idx = np.argsort(query_lis_score)[-top_n:]
        return lis_idx


bm25_ranking = Bm25PreRanking(cached_folder_path='bm25_ranking/cached')

if __name__ == '__main__':
    with open('segemented_data/segmented_corpus.json', 'r') as f:
        legal_corpus = json.load(f)

    with open('segemented_data/train_ques_segmented.json', 'r') as f:
        ques_corpus = json.load(f)

    with open('segemented_data/test_ques_segmented.json', 'r') as f:
        test_ques_corpus = json.load(f)

    test_query_ids = 0
    # print(' '.join([tok for sent in ques_corpus[test_query_ids] for tok in sent]))
    # ranking_ids = bm25_ranking.get_ranking(query_idx=test_query_ids, prefix='train_ques', top_n=5)

    print(' '.join([tok for sent in test_ques_corpus[test_query_ids] for tok in sent]))
    ranking_ids = bm25_ranking.get_ranking(query_idx=test_query_ids, prefix='test_ques', top_n=5)

    for ids in ranking_ids:
        article = legal_corpus[ids]
        print(' '.join([tok for sent in article for tok in sent]))
