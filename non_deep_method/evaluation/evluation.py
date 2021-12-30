from tqdm import tqdm

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
        self.top_n = 50
        self.clf = clf

    def predict_lis_article(self, query_idx, lis_article_idx):
        x = np.array([
            self.xy_data_builder.train_q_corpus.get_features(ques_idx=query_idx, corpus_idx=article_idx)
            for article_idx in lis_article_idx
        ])
        lis_prob = self.clf.predict_proba(X=x)
        lis_prob = lis_prob[:, 1]
        return lis_prob

    def cal_candidate_prob(self, query_idx):
        candidate_article = self.bm25ranking.get_ranking(query_idx=query_idx, prefix='train_ques', top_n=self.top_n)
        prob_article = self.predict_lis_article(query_idx, candidate_article)
        return candidate_article, prob_article

    def start_evaluate_f2i_score(self):
        truth_candidate_prob = []
        for query_idx in self.lis_idx:
            ground_truth_article = self.xy_data_builder.find_relevance_i_article(ques_id=query_idx, prefix='train_ques')
            candidate_article, prob_article = self.cal_candidate_prob(query_idx)
            truth_candidate_prob.append((ground_truth_article, candidate_article, prob_article))

        max_f2score = 0
        best_threshold = None
        for threshold in np.arange(0, 1, 0.001):
            sum_f2score = 0
            for ground_truth_article, candidate_article, prob_article in tqdm(truth_candidate_prob,
                                                                              desc='Start inference'):
                pred_article = [candidate_article[i] for i, prob in enumerate(prob_article) if prob >= threshold]
                sum_f2score += calculate_f2i(lis_ground_truth=ground_truth_article, lis_predict=pred_article)
            if sum_f2score > max_f2score:
                max_f2score = sum_f2score
                best_threshold = threshold

        print('Max f2score: ', max_f2score / len(truth_candidate_prob))
        print('Best threshold: ', best_threshold)


if __name__ == '__main__':
    evaluation_machine = Evaluation(clf=model_builder.clf)
    evaluation_machine.start_evaluate_f2i_score()
