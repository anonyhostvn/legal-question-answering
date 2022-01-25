import numpy as np

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


class DataBuilder:
    def __init__(self, legal_ques_corpus: LegalQuestionCorpus, legal_corpus: LegalCorpus):
        self.legal_ques_corpus = legal_ques_corpus
        self.legal_corpus = legal_corpus

        self.pairwise_uni_tfidf = PairwiseTfidf(self.legal_corpus.uni_corpus, self.legal_ques_corpus.uni_corpus)
        self.pairwise_bi_tfidf = PairwiseTfidf(self.legal_corpus.bi_corpus, self.legal_ques_corpus.bi_corpus)

        self.ftext_machine = ftext_machine

    def get_feature_vector(self, ques_idx, corpus_idx):
        cos_uni_tfidf = self.pairwise_uni_tfidf.get_cosine_sim(ques_idx, corpus_idx)
        cos_bi_tfidf = self.pairwise_bi_tfidf.get_cosine_sim(ques_idx, corpus_idx)
        ques_uni_tfidf = self.pairwise_uni_tfidf.ques_tfidf[ques_idx]
        legal_uni_tfidf = self.pairwise_uni_tfidf.legal_tfidf[corpus_idx]
        ques_wemb = self.ftext_machine.get_wavg_word_emb(tfidf_score=ques_uni_tfidf,
                                                         vocab=self.pairwise_uni_tfidf.get_vocab())
        corpus_wemb = self.ftext_machine.get_wavg_word_emb(tfidf_score=legal_uni_tfidf,
                                                           vocab=self.pairwise_uni_tfidf.get_vocab())

        return np.concatenate(([cos_uni_tfidf, cos_bi_tfidf], ques_wemb, corpus_wemb), axis=0)