from utilities.utils import transform_seg2uni, transform_seg2bi
from abc import ABCMeta, abstractmethod


class AbstractCorpus(metaclass=ABCMeta):
    def __init__(self, json_corpus, seg_corpus):
        super().__init__()
        self.json_corpus = json_corpus
        self.seg_corpus = seg_corpus

        self.uni_corpus = transform_seg2uni(self.seg_corpus)
        self.bi_corpus = transform_seg2bi(self.seg_corpus)

    def get_uni_corpus(self, idx):
        return self.uni_corpus[idx]

    @abstractmethod
    def get_len_corpus(self):
        pass
