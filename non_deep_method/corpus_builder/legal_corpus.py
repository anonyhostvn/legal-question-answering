import json

from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS
from non_deep_method.config import LOAD_SAMPLE_SIZE
from non_deep_method.corpus_builder.abstract_corpus import AbstractCorpus


class LegalCorpus(AbstractCorpus):
    def __init__(self, corpus_json_path, corpus_segmented_path, sample_size=None):
        with open(corpus_json_path, 'r') as f:
            self.json_corpus = json.load(f)[:sample_size]

        with open(corpus_segmented_path, 'r') as f:
            self.segmented_corpus = json.load(f)[:sample_size]

        super().__init__(self.json_corpus, self.segmented_corpus)


if __name__ == '__main__':
    legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH,
                               corpus_segmented_path=SEGMENTED_LEGAL_CORPUS, sample_size=LOAD_SAMPLE_SIZE)
