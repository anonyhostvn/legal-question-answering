from tqdm import tqdm

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import TRAIN_IDX, TEST_IDX, DATA_QUESTION_PATH, LEGAL_CORPUS_PATH, TEST_QUESTION_PATH
import json

from non_deep_method.config import CACHE_DIR
from non_deep_method.corpus_builder.question_corpus import train_q_corpus
import numpy as np
import os


class XYDataBuilder:
    def __init__(self):
        with open(DATA_QUESTION_PATH, 'r') as f:
            data_question = json.load(f).get('items')

        with open(TEST_QUESTION_PATH, 'r') as f:
            test_question = json.load(f).get('items')

        self.question_cluster = {
            'train_ques': data_question,
            'test_ques': test_question
        }

        with open(LEGAL_CORPUS_PATH, 'r') as f:
            self.legal_corpus = json.load(f)
            self.lis_legal_article = [
                {**article, 'law_id': legal.get('law_id')} for legal in self.legal_corpus
                for article in legal.get('articles')]

        self.train_q_corpus = train_q_corpus

        self.bm25_ranking = bm25_ranking

        self.get_cached_filename = lambda top_n, phase, xy: os.path.join(CACHE_DIR, f'{phase}-top_n_{top_n}-{xy}.npy')

    def find_i_article(self, law_id, article_id):
        for i_article, article in enumerate(self.lis_legal_article):
            if article.get('law_id') == law_id and article.get('article_id') == article_id:
                return i_article

    def find_relevance_i_article(self, ques_id, prefix):
        assert prefix in self.question_cluster.keys(), 'prefix cluster is not exist'
        data_question = self.question_cluster.get(prefix)
        lis_i_article = []
        for relevance_article in data_question[ques_id].get('relevant_articles'):
            lis_i_article.append(
                self.find_i_article(relevance_article.get('law_id'), relevance_article.get('article_id'))
            )
        return lis_i_article

    def build_data_with_features(self, top_n, phase, prefix):
        x_cached_filename = self.get_cached_filename(top_n=top_n, phase=phase, xy='X')
        y_cached_filename = self.get_cached_filename(top_n=top_n, phase=phase, xy='y')
        if os.path.exists(x_cached_filename) and os.path.exists(y_cached_filename):
            return np.load(x_cached_filename), np.load(y_cached_filename)

        if phase == 'train_phase':
            with open(TRAIN_IDX, 'r') as f:
                lis_ques_idx = json.load(f)
        elif phase == 'test_phase':
            with open(TEST_IDX, 'r') as f:
                lis_ques_idx = json.load(f)

        X = []
        y = []
        for ques_idx in tqdm(lis_ques_idx):
            lis_i_relevance_article = set(self.find_relevance_i_article(ques_id=ques_idx, prefix=prefix))
            top_n_relevance_article = set(self.bm25_ranking.get_ranking(query_idx=ques_idx, prefix=prefix, top_n=top_n))
            top_n_relevance_article.update(lis_i_relevance_article)
            for i_relevance in top_n_relevance_article:
                X.append(self.train_q_corpus.get_features(ques_idx=ques_idx, corpus_idx=i_relevance))
                y.append(int(i_relevance in lis_i_relevance_article))
        X = np.array(X)
        y = np.array(y)

        np.save(file=x_cached_filename, arr=X)
        np.save(file=y_cached_filename, arr=y)
        return X, y


data_builder = XYDataBuilder()

if __name__ == '__main__':
    X_test, y_test = data_builder.build_data_with_features(top_n=50, phase='test_phase', prefix='train_ques')
    X_train, y_train = data_builder.build_data_with_features(top_n=50, phase='train_phase', prefix='train_ques')
    print(X_train.shape)
    print(X_test.shape)
