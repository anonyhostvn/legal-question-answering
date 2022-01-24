from bm25_ranking.bm25_pre_ranking import bm25_ranking
import json

from global_config import LEGAL_CORPUS_PATH, DATA_QUESTION_PATH
from non_deep_method.evaluation.eval_utils import calculate_recall
from tqdm import tqdm


class Bm25Eval:
    def __init__(self):
        self.bm25ranking = bm25_ranking
        with open(LEGAL_CORPUS_PATH, 'r', encoding='utf-8') as f:
            self.legal_corpus = json.load(f)
            self.legal_article_id = [(law.get('law_id'), legal_article.get('article_id')) for law in self.legal_corpus
                                     for legal_article in law.get('articles')]

            self.legal_article_mapping = {
                law.get('law_id'): {
                    single_legal_article_id[1]: aidx
                    for aidx, single_legal_article_id in enumerate(self.legal_article_id) if
                    single_legal_article_id[0] == law.get('law_id')
                } for law in self.legal_corpus
            }

    def start_eval(self, json_data_path, top_n=50):
        with open(json_data_path, 'r', encoding='utf-8') as f:
            ques_corpus = json.load(f).get('items')

        sum_f2score = 0
        for qidx, single_ques in enumerate(tqdm(ques_corpus)):
            lis_predict_aidx = self.bm25ranking.get_ranking(query_idx=qidx, prefix='train_ques', top_n=top_n)
            lis_ground_truth_aidx = []
            for single_relevance_article in single_ques['relevant_articles']:
                g_law_id = single_relevance_article.get('law_id')
                g_art_id = single_relevance_article.get('article_id')
                lis_ground_truth_aidx.append(self.legal_article_mapping[g_law_id][g_art_id])
            sum_f2score += calculate_recall(lis_ground_truth=lis_ground_truth_aidx, lis_predict=lis_predict_aidx)
        print('Average recall: ', sum_f2score / len(ques_corpus))


if __name__ == '__main__':
    bm25eval = Bm25Eval()
    bm25eval.start_eval(json_data_path=DATA_QUESTION_PATH)
