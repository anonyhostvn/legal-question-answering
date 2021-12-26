import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from non_deep_method.config import CACHE_DIR
from non_deep_method.legal_corpus import LegalCorpus
from non_deep_method.utils import transform_seg2bi, transform_seg2uni, get_wavg_word_emb_with_cached
from utils import jaccard_similarity


class QuestionCorpus:
    def __init__(self, ques_json_path: str, ques_segmented_path: str,
                 legal_corpus: LegalCorpus, phase: str,
                 sample_size: int = None) -> None:
        self.phase = phase
        self.legal_corpus = legal_corpus

        with open(ques_json_path, 'r') as f:
            self.ques = json.load(f).get('items')[:sample_size]
        with open(ques_segmented_path, 'r') as f:
            self.segmented_ques = json.load(f)[:sample_size]

        self.bi_ques = transform_seg2bi(self.segmented_ques)
        self.uni_ques = transform_seg2uni(self.segmented_ques)
        self.ques_uni_tfidf = legal_corpus.uni_tfidf.vectorizer.transform(self.uni_ques)
        self.uni_vocab = legal_corpus.uni_vocab
        self.uni_tfidf_cosine = cosine_similarity(X=self.ques_uni_tfidf,
                                                  Y=self.legal_corpus.uni_tfidf.tf_idf_corpus,
                                                  dense_output=False)

        self.ques_bi_tfidf = legal_corpus.bi_tfidf.vectorizer.transform(self.bi_ques)
        self.bi_tfidf_cosine = cosine_similarity(X=self.ques_bi_tfidf,
                                                 Y=self.legal_corpus.bi_tfidf.tf_idf_corpus,
                                                 dense_output=False)

    def get_w_avg_word_emb(self, idx: int):
        cached_filename = os.path.join(CACHE_DIR, f'{self.phase}_ques_{idx}.wavg_emb.npy')
        return get_wavg_word_emb_with_cached(tfidf_score=self.ques_uni_tfidf[idx], vocab=self.uni_vocab,
                                             cached_filename=cached_filename)

    def get_features(self, ques_idx, corpus_idx):
        cosine_uni_tfidf = self.uni_tfidf_cosine[ques_idx][corpus_idx]
        cosine_bi_tfidf = self.bi_tfidf_cosine[ques_idx][corpus_idx]
        j_score = jaccard_similarity(self.uni_ques[ques_idx].split(' '),
                                     self.legal_corpus.uni_corpus[corpus_idx].split(' '))
        basic_features = np.array([cosine_uni_tfidf, cosine_bi_tfidf, j_score])
        word_embed_features = np.concatenate(
            (self.get_w_avg_word_emb(ques_idx),
             self.legal_corpus.get_w_avg_word_emb(corpus_idx)), dim=0)
        combine_features = np.concatenate((basic_features, word_embed_features), dim=0)
        return combine_features


if __name__ == '__main__':
    l_corpus = LegalCorpus(
        corpus_json_path='/Users/LongNH/Workspace/ZaloAIChallenge/zac2021-ltr-data/legal_corpus.json',
        corpus_segmented_path='/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/segmented_corpus.json',
        sample_size=100
    )

    q_corpus = QuestionCorpus(
        ques_json_path='/Users/LongNH/Workspace/ZaloAIChallenge/zac2021-ltr-data/train_question_answer.json',
        ques_segmented_path='/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/train_ques_segmented.json',
        phase='train',
        sample_size=100,
        legal_corpus=l_corpus
    )

    print(q_corpus.get_features(ques_idx=0, corpus_idx=0).shape)
