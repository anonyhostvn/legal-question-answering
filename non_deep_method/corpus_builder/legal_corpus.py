import json
import os

from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS
from non_deep_method.config import LOAD_SAMPLE_SIZE, CACHE_DIR
from non_deep_method.utilities.tfidf_machine import TfIdfMachine
from non_deep_method.utilities.utils import transform_seg2uni, transform_seg2bi, get_wavg_word_emb_with_cached


class LegalCorpus:
    def __init__(self, corpus_json_path, corpus_segmented_path, sample_size=None):
        self.bi_corpus = None
        self.uni_corpus = None
        with open(corpus_json_path, 'r') as f:
            self.corpus = json.load(f)[:sample_size]

        with open(corpus_segmented_path, 'r') as f:
            self.segmented_corpus = json.load(f)[:sample_size]

        print('\n building uni-gram tfidf ... ')
        self.uni_corpus = transform_seg2uni(self.segmented_corpus)
        self.uni_tfidf = TfIdfMachine(self.uni_corpus)
        self.uni_vocab = self.uni_tfidf.vectorizer.get_feature_names_out()

        print('\n building bi-gram tfidf ...')
        self.bi_corpus = transform_seg2bi(self.segmented_corpus)
        self.bi_tfidf = TfIdfMachine(self.bi_corpus)

    def get_w_avg_word_emb(self, idx: int):
        cached_filename = os.path.join(CACHE_DIR, f'legal_{idx}.wavg_emb.npy')
        return get_wavg_word_emb_with_cached(tfidf_score=self.uni_tfidf.get_tfidf(idx), vocab=self.uni_vocab,
                                             cached_filename=cached_filename)


if __name__ == '__main__':
    legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH,
                               corpus_segmented_path=SEGMENTED_LEGAL_CORPUS, sample_size=LOAD_SAMPLE_SIZE)
