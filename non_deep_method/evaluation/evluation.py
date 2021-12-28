from bm25_ranking.bm25_pre_ranking import bm25_ranking
from lightgbm import LGBMClassifier
from global_config import TEST_IDX
import json
import numpy as np

from non_deep_method.data_builder.xy_data_builder import data_builder
from non_deep_method.evaluation.eval_utils import calculate_f2i
from non_deep_method.xgb_model.model_builder import model_builder


class Evaluation:
    def __init__(self, clf: LGBMClassifier):
        with open(TEST_IDX, 'r') as f:
            self.lis_idx = json.load(f)

        self.bm25ranking = bm25_ranking

        self.xy_data_builder = data_builder
        self.threshold = 0.1
        self.top_n = 50
        self.clf = clf

    def predict_lis_article(self, query_idx, lis_article_idx):
        x = np.array([
            self.xy_data_builder.train_q_corpus.get_features(ques_idx=query_idx, corpus_idx=article_idx)
            for article_idx in lis_article_idx
        ])
        lis_prob = self.clf.predict_proba(X=x)
        lis_prob = lis_prob[:, 1]
        return [i for i, prob in enumerate(lis_prob) if prob >= self.threshold]

    def calculate_single_f2i(self, query_idx):
        candidate_article = self.bm25ranking.get_ranking(query_idx=query_idx, prefix='train_ques', top_n=self.top_n)
        ground_truth_article = self.xy_data_builder.find_relevance_i_article(ques_id=query_idx, prefix='train_ques')
        predict_article = self.predict_lis_article(query_idx, candidate_article)
        return calculate_f2i(lis_predict=predict_article, lis_ground_truth=ground_truth_article)

    def start_evaluate_f2i_score(self):
        sum_f2i = 0
        for query_idx in self.lis_idx:
            sum_f2i += self.calculate_single_f2i(query_idx)
        print('Average F2i score: ', sum_f2i / len(self.lis_idx))


if __name__ == '__main__':
    evaluation_machine = Evaluation(clf=model_builder.clf)
    evaluation_machine.start_evaluate_f2i_score()
