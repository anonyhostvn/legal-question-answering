from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, TEST_IDX
import json

from utilities.utils import transform_seg2uni


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

    def generate_data_for_semantic_sim(self):
        for corpus_id in range(len(self.segmented_corpus)):
            
        pass

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
                                     test_idx_path=TEST_IDX)
    data_preprocess.test_bm25()
