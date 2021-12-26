import json

from non_deep_method.config import LOAD_SAMPLE_SIZE
from non_deep_method.tfidf_machine import TfIdfMachine
from tqdm import tqdm

from non_deep_method.utils import transform_seg2uni, transform_seg2bi


class LegalCorpus:
    def __init__(self, corpus_json_path, corpus_segmented_path, sample_size=None):
        self.bi_corpus = None
        self.uni_corpus = None
        with open(corpus_json_path, 'r') as f:
            self.corpus = json.load(f)[:sample_size]

        with open(corpus_segmented_path, 'r') as f:
            self.segmented_corpus = json.load(f)[:sample_size]

        print('\n building uni-gram tfidf ... ')
        self.uni_tfidf = TfIdfMachine(transform_seg2uni(self.segmented_corpus))
        print('\n building bi-gram tfidf ...')
        self.bi_tfidf = TfIdfMachine(transform_seg2bi(self.segmented_corpus))


if __name__ == '__main__':
    CORPUS_PATH = '/Users/LongNH/Workspace/ZaloAIChallenge/zac2021-ltr-data/legal_corpus.json'
    SEGMENTED_CORPUS_PATH = '/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/segmented_corpus.json'

    legal_corpus = LegalCorpus(corpus_json_path=CORPUS_PATH,
                               corpus_segmented_path=SEGMENTED_CORPUS_PATH, sample_size=LOAD_SAMPLE_SIZE)
