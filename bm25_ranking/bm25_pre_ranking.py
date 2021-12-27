import os

from rank_bm25 import BM25Okapi
import json

from global_config import SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, SEGMENTED_TEST_QUESTION
import numpy as np
from tqdm import tqdm


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

    def get_top_n(self, tokenized_query, n):
        lis_score = self.bm25.get_scores(tokenized_query)
        top_n_idx = np.argsort(lis_score)[:n]
        return top_n_idx

    def pre_score_calculated(self, prefix):
        assert prefix in self.cluster_segmented_query.keys(), 'Prefix is not in segmented database'
        print('Start compute ranking score')

        with open(SEGMENTED_LEGAL_CORPUS, 'r') as f:
            segmented_legal_corpus = json.load(f)

        modified_segmented_legal_corpus = [[tok for sent in article for tok in sent]
                                           for article in tqdm(segmented_legal_corpus)]
        self.bm25 = BM25Okapi(modified_segmented_legal_corpus)

        with open(self.cluster_segmented_query.get(prefix), 'r') as f:
            lis_segmented_query = json.load(f)

        lis_score = []
        for segmented_query in lis_segmented_query:
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


if __name__ == '__main__':
    bm25_ranking = Bm25PreRanking(cached_folder_path='/Users/LongNH/Workspace/ZaloAIChallenge/bm25_ranking/cached')

    with open('/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/segmented_corpus.json', 'r') as f:
        legal_corpus = json.load(f)

    with open('/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/train_ques_segmented.json', 'r') as f:
        ques_corpus = json.load(f)

    test_query_ids = 0
    print(' '.join([tok for sent in ques_corpus[test_query_ids] for tok in sent]))
    ranking_ids = bm25_ranking.get_ranking(query_idx=test_query_ids, prefix='train_ques', top_n=5)

    for ids in ranking_ids:
        article = legal_corpus[ids]
        print(' '.join([tok for sent in article for tok in sent]))
