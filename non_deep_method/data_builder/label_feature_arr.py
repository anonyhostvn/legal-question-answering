from tqdm import tqdm

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import TRAIN_IDX, TEST_IDX, DATA_QUESTION_PATH, LEGAL_CORPUS_PATH
import json

from non_deep_method.config import CACHE_DIR
from non_deep_method.corpus_builder.question_corpus import train_q_corpus
import numpy as np
import os


class XYDataBuilder:
    def __init__(self):
        with open(DATA_QUESTION_PATH, 'r') as f:
            self.data_question = json.load(f).get('items')
        with open(LEGAL_CORPUS_PATH, 'r') as f:
            self.legal_corpus = json.load(f)
            self.lis_legal_article = [
                {**article, 'law_id': legal.get('law_id')} for legal in self.legal_corpus
                for article in legal.get('articles')]

        self.train_q_corpus = train_q_corpus

        self.bm25_ranking = bm25_ranking

        self.get_cached_filename = lambda top_n, prefix, xy: os.path.join(CACHE_DIR, f'{prefix}-top_n_{top_n}-{xy}.npy')

    def find_i_article(self, law_id, article_id):
        for i_article, article in enumerate(self.lis_legal_article):
            if article.get('law_id') == law_id and article.get('article_id') == article_id:
                return i_article

    def find_relevance_i_article(self, ques_id):
        lis_i_article = []
        for relevance_article in self.data_question[ques_id].get('relevant_articles'):
            lis_i_article.append(
                self.find_i_article(relevance_article.get('law_id'), relevance_article.get('article_id'))
            )
        return lis_i_article

    def build_data_with_features(self, top_n, prefix):
        x_cached_filename = self.get_cached_filename(top_n=top_n, prefix=prefix, xy='X')
        y_cached_filename = self.get_cached_filename(top_n=top_n, prefix=prefix, xy='y')
        if os.path.exists(x_cached_filename) and os.path.exists(y_cached_filename):
            X = np.load(x_cached_filename)
            y = np.load(y_cached_filename)
            return X, y

        if prefix == 'train_ques':
            with open(TRAIN_IDX, 'r') as f:
                lis_ques_idx = json.load(f)
        elif prefix == 'test_ques':
            with open(TEST_IDX, 'r') as f:
                lis_ques_idx = json.load(f)

        X = []
        y = []
        for ques_idx in tqdm(lis_ques_idx):
            lis_i_relevance_article = set(self.find_relevance_i_article(ques_idx))
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


if __name__ == '__main__':
    data_builder = XYDataBuilder()
    X, y = data_builder.build_data_with_features(top_n=50, prefix='train_ques')
    print(X.shape)
