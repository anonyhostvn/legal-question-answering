from utilities.utils import transform_seg2uni, transform_seg2bi


class AbstractCorpus:
    def __init__(self, json_corpus, seg_corpus):
        super().__init__()
        self.json_corpus = json_corpus
        self.seg_corpus = seg_corpus

        self.uni_corpus = transform_seg2uni(self.seg_corpus)
        self.bi_corpus = transform_seg2bi(self.seg_corpus)
