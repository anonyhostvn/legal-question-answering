import json

from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS
from non_deep_method.corpus_builder.abstract_corpus import AbstractCorpus


class LegalCorpus(AbstractCorpus):
    def __init__(self, corpus_json_path, corpus_segmented_path):
        with open(corpus_json_path, 'r') as f:
            self.json_corpus = json.load(f)
            self.lis_article = [
                {**article_obj, 'law_id': law_obj.get('law_id')} for law_obj in self.json_corpus for article_obj in
                law_obj.get('articles')
            ]

        with open(corpus_segmented_path, 'r') as f:
            self.segmented_corpus = json.load(f)

        super().__init__(self.json_corpus, self.segmented_corpus)

    def get_aidx(self, law_id, article_id):
        for aidx, article_obj in enumerate(self.lis_article):
            if article_obj.get('law_id') == law_id and article_obj.get('article_id') == article_id:
                return aidx


if __name__ == '__main__':
    legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH,
                               corpus_segmented_path=SEGMENTED_LEGAL_CORPUS)
