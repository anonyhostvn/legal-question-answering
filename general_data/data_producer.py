import json

from global_config import TRAIN_IDX, SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, DATA_QUESTION_PATH, \
    LEGAL_CORPUS_PATH, TEST_IDX


class DataProducer:
    def __init__(self):
        with open(TRAIN_IDX, 'r') as f:
            self.lis_train_idx = json.load(f)

        with open(TEST_IDX, 'r') as f:
            self.lis_test_idx = json.load(f)

        with open(SEGMENTED_LEGAL_CORPUS, 'r', encoding='utf-8') as f:
            self.segmented_legal_corpus = json.load(f)

        with open(SEGMENTED_DATA_QUESTION, 'r', encoding='utf-8') as f:
            self.segmented_ques = json.load(f)

        with open(DATA_QUESTION_PATH, 'r', encoding='utf-8') as f:
            self.dat_ques = json.load(f).get('items')

        with open(LEGAL_CORPUS_PATH, 'r', encoding='utf-8') as f:
            self.legal_corpus = json.load(f)
            self.lis_legal_article = [
                {**article, 'law_id': legal.get('law_id')} for legal in self.legal_corpus
                for article in legal.get('articles')]

    def get_segmented_legal_article(self, aid):
        return self.segmented_legal_corpus[aid]

    def get_segmented_ques(self, qid):
        return self.segmented_ques[qid]

    def find_article_idx(self, law_id, art_id):
        for idx, legal_article in enumerate(self.lis_legal_article):
            if law_id == legal_article.get('law_id') and art_id == legal_article.get('article_id'):
                return idx
        return None

    def get_ground_truth_relevance_article_ques(self, qid):
        rec_ques = self.dat_ques[qid]
        lis_ground_truth_relevance_article = []
        for rel_info in rec_ques.get('relevant_articles'):
            law_id = rel_info.get('law_id')
            art_id = rel_info.get('article_id')
            lis_ground_truth_relevance_article.append(self.find_article_idx(law_id, art_id))
        return lis_ground_truth_relevance_article
