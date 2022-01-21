import json
import os

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH, TEST_QUESTION_PATH, SEGMENTED_TEST_QUESTION, \
    LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS
from tqdm import tqdm
from sent_bert_training.sent_bert_downstream import SentBertDownstream


class DownstreamInference:
    def __init__(self):
        self.model = SentBertDownstream(pretrain_sent_bert=SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH)
        self.bm25 = bm25_ranking
        with open(SEGMENTED_LEGAL_CORPUS, 'r', encoding='utf-8') as f:
            self.segmented_corpus = json.load(f)
        with open(LEGAL_CORPUS_PATH, 'r', encoding='utf-8') as f:
            self.legal_corpus = json.load(f)
            self.legal_article_id = [(law.get('law_id'), legal_article.get('article_id')) for law in self.legal_corpus
                                     for legal_article in law.get('articles')]

    def calculate_ranking_score_sent_bert(self, txt_ques, lis_txt_article):
        return self.model.inferences(sent_1=txt_ques, lis_sent_2=lis_txt_article)

    def predict_query(self, segmented_query, lis_bm25_ranking, threshold=0.9):
        txt_query = ' '.join([tok for sent in segmented_query for tok in sent])
        lis_txt_articles = [' '.join([tok for sent in self.segmented_corpus[aidx] for tok in sent])
                            for aidx in lis_bm25_ranking]
        lis_ranking_score = self.calculate_ranking_score_sent_bert(txt_query, lis_txt_articles)
        lis_aidx = []
        for i, ranking_score in enumerate(lis_ranking_score):
            if ranking_score >= threshold:
                lis_aidx.append(lis_bm25_ranking[i])
        return lis_aidx

    def inference_public_test(self):
        with open(TEST_QUESTION_PATH, 'r', encoding='utf-8') as f:
            lis_test_ques = json.load(f).get('items')
        with open(SEGMENTED_TEST_QUESTION, 'r', encoding='utf-8') as f:
            lis_segmented_test_ques = json.load(f)

        public_test_submission = []
        for ques_idx, segmented_ques in enumerate(tqdm(lis_segmented_test_ques)):
            lis_bm25_ranking = self.bm25.get_ranking(query_idx=ques_idx, prefix='test_ques', top_n=100)
            lis_aidx = self.predict_query(segmented_ques, lis_bm25_ranking)
            relevance_articles = [
                {'law_id': self.legal_article_id[aidx][0], 'article_id': self.legal_article_id[aidx][1]}
                for aidx in lis_aidx]

            public_test_submission.append({
                'question_id': lis_test_ques[ques_idx].get('question_id'),
                'relevant_articles': relevance_articles
            })

        with open(os.path.join('submission_folder', 'public_test_submission.json'), 'w') as f:
            json.dump(public_test_submission, f)


if __name__ == '__main__':
    downstream_inference = DownstreamInference()
    downstream_inference.inference_public_test()
