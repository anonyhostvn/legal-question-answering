import json
import numpy as np
from non_deep_method.legal_corpus import LegalCorpus
from non_deep_method.utils import transform_seg2bi, transform_seg2uni
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


class QuestionCorpus:
    def __init__(self, ques_json_path: str, ques_segmented_path: str, legal_corpus: LegalCorpus,
                 sample_size: int = None) -> None:
        with open(ques_json_path, 'r') as f:
            self.ques = json.load(f)[:sample_size]
        with open(ques_segmented_path, 'r') as f:
            self.segmented_ques = json.load(f)[:sample_size]

        self.bi_ques = transform_seg2bi(self.segmented_ques)
        self.uni_ques = transform_seg2uni(self.segmented_ques)
        self.ques_uni_tfidf = legal_corpus.uni_tfidf.vectorizer.transform(self.uni_ques)
        self.ques_bi_tfidf = legal_corpus.bi_tfidf.vectorizer.transform(self.bi_ques)
        self.legal_corpus = legal_corpus

        self.uni_tfidf_cosine = cosine_similarity(X=self.ques_uni_tfidf,
                                                  Y=self.legal_corpus.uni_tfidf,
                                                  dense_output=False)
        self.bi_tfidf_cosine = cosine_similarity(X=self.ques_bi_tfidf,
                                                 Y=self.legal_corpus.bi_tfidf,
                                                 dense_output=False)

    def get_features(self, ques_idx, corpus_idx):
        cosine_uni_tfidf = self.uni_tfidf_cosine[ques_idx][corpus_idx]
        cosine_bi_tfidf = self.bi_tfidf_cosine[ques_idx][corpus_idx]
        j_score = jaccard_score(self.uni_ques[ques_idx], self.legal_corpus.uni_corpus[corpus_idx])
        return np.array([cosine_uni_tfidf, cosine_bi_tfidf, j_score])
