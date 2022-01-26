import random

import numpy as np

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import DATA_QUESTION_PATH, SEGMENTED_DATA_QUESTION, LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS
from non_deep_method.corpus_builder.ftext_machine import ftext_machine
from non_deep_method.corpus_builder.legal_corpus import LegalCorpus
from non_deep_method.corpus_builder.legal_question_corpus import LegalQuestionCorpus
from utilities.tfidf_machine import TfIdfMachine
from sklearn.metrics.pairwise import cosine_similarity


class PairwiseTfidf:
    def __init__(self, legal_corpus, ques_corpus):
        self.tf_idf_machine = TfIdfMachine(corpus=legal_corpus)
        self.legal_tfidf = self.tf_idf_machine.get_transformed_root_corpus()
        self.ques_tfidf = self.tf_idf_machine.transform_corpus(ques_corpus)
        self.cosine_sim = cosine_similarity(X=self.ques_tfidf, Y=self.legal_tfidf)

    def get_cosine_sim(self, ques_idx, corpus_idx):
        return self.cosine_sim[ques_idx, corpus_idx]

    def get_vocab(self):
        return self.tf_idf_machine.vectorizer.get_feature_names_out()


def calculate_jaccard_sim(x, y):
    cnt = 0
    for ix in x:
        if ix in y:
            cnt += 1
    return cnt / (len(x) + len(y) - cnt)


class DataBuilder:
    def __init__(self, legal_ques_corpus: LegalQuestionCorpus, legal_corpus: LegalCorpus):
        self.legal_ques_corpus = legal_ques_corpus
        self.legal_corpus = legal_corpus

        self.pairwise_uni_tfidf = PairwiseTfidf(legal_corpus=self.legal_corpus.uni_corpus,
                                                ques_corpus=self.legal_ques_corpus.uni_corpus)
        self.pairwise_bi_tfidf = PairwiseTfidf(legal_corpus=self.legal_corpus.bi_corpus,
                                               ques_corpus=self.legal_ques_corpus.bi_corpus)

        self.ftext_machine = ftext_machine
        self.bm25ranking = bm25_ranking

    def get_relevant_aidx_of_ques(self, qidx):
        lis_relevant_article = self.legal_ques_corpus.get_lis_relevant_article(qidx)
        lis_relevant_aidx = [self.legal_corpus.get_aidx(article.get('law_id'), article.get('article_id'))
                             for article in lis_relevant_article]
        return lis_relevant_aidx

    def get_random_aidx(self):
        return random.randint(0, self.legal_corpus.get_len_corpus() - 1)

    def get_random_non_relevant_aidx_of_ques(self, lis_relevant_aidx):
        random_aidx = self.get_random_aidx()
        while random_aidx in lis_relevant_aidx:
            random_aidx = self.get_random_aidx()
        return random_aidx

    def get_bm25_ranking(self, qidx, top_n):
        return self.bm25ranking.get_ranking(query_idx=qidx, prefix='train_ques', top_n=top_n)

    def get_non_relevant_aidx_of_ques_bm25(self, qidx, lis_relevant_aidx, n_elements):
        lis_bm25_ranking = self.get_bm25_ranking(qidx, top_n=n_elements * 2)
        lis_aidx = []
        for aidx in lis_bm25_ranking:
            if aidx not in lis_relevant_aidx:
                lis_aidx.append(aidx)
            if len(lis_aidx) >= n_elements:
                break
        return lis_aidx

    def get_feature_vector(self, ques_idx, corpus_idx):
        cos_uni_tfidf = self.pairwise_uni_tfidf.get_cosine_sim(ques_idx, corpus_idx)
        cos_bi_tfidf = self.pairwise_bi_tfidf.get_cosine_sim(ques_idx, corpus_idx)
        ques_uni_tfidf = self.pairwise_uni_tfidf.ques_tfidf[ques_idx]
        legal_uni_tfidf = self.pairwise_uni_tfidf.legal_tfidf[corpus_idx]
        ques_wemb = self.ftext_machine.get_wavg_word_emb(tfidf_score=ques_uni_tfidf,
                                                         vocab=self.pairwise_uni_tfidf.get_vocab())
        corpus_wemb = self.ftext_machine.get_wavg_word_emb(tfidf_score=legal_uni_tfidf,
                                                           vocab=self.pairwise_uni_tfidf.get_vocab())
        jaccard_sim = calculate_jaccard_sim(self.legal_ques_corpus.get_uni_corpus(ques_idx).split(' '),
                                            self.legal_corpus.get_uni_corpus(corpus_idx).split(' '))

        return np.concatenate(([jaccard_sim, cos_uni_tfidf, cos_bi_tfidf], ques_wemb, corpus_wemb), axis=0)


if __name__ == '__main__':
    train_ques_corpus = LegalQuestionCorpus(json_ques_path=DATA_QUESTION_PATH, seg_ques_path=SEGMENTED_DATA_QUESTION)
    comp_legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH, corpus_segmented_path=SEGMENTED_LEGAL_CORPUS)

    data_builder = DataBuilder(train_ques_corpus, comp_legal_corpus)
    data_builder.get_feature_vector(ques_idx=0, corpus_idx=0)
    print('Done !')
