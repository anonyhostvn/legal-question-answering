from sentence_transformers import SentenceTransformer, util
import json

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, LEGAL_CORPUS_PATH, DATA_QUESTION_PATH, \
    TEST_IDX, SENT_BERT_CHECKPOINT
from non_deep_method.evaluation.eval_utils import calculate_f2i
from utilities.utils import transform_seg2uni
from tqdm import tqdm


class Evaluation:
    def __init__(self, sent_bert_model: SentenceTransformer, test_idx_path: str, segmented_corpus_path: str,
                 segmented_ques_path: str, ques_json_path: str, corpus_json_path: str):
        self.sent_bert_model = sent_bert_model

        with open(test_idx_path, 'r') as f:
            self.lis_test_idx = json.load(f)

        with open(segmented_corpus_path, 'r') as f:
            self.segmented_corpus = json.load(f)
        with open(segmented_ques_path, 'r') as f:
            self.segmented_ques = json.load(f)

        with open(ques_json_path, 'r', encoding='utf-8') as f:
            self.lis_ques = json.load(f).get('items')
        with open(corpus_json_path, 'r', encoding='utf-8') as f:
            self.lis_corpus = json.load(f)
            self.lis_corpus = [{'law_id': law.get('law_id'), **article}
                               for law in self.lis_corpus for article in law.get('articles')]
        self.bm25_ranking = bm25_ranking

    def find_aid(self, law_id, article_id):
        for aid, article in enumerate(self.lis_corpus):
            if article.get('law_id') == law_id and article.get('article_id') == article_id:
                return aid

    def get_list_relevance_article(self, lis_ques):
        lis_aid = []
        for ques_id in lis_ques:
            rel_aid = []
            for article in self.lis_ques[ques_id].get('relevant_articles'):
                law_id = article.get('law_id')
                article_id = article.get('article_id')
                rel_aid.append(self.find_aid(law_id, article_id))
            lis_aid.append(rel_aid)
        return lis_aid

    def start_evaluate(self, top_n, threshold=0.8):
        lis_label = self.get_list_relevance_article(self.lis_test_idx)

        segmented_test_ques = [self.segmented_ques[i] for i in self.lis_test_idx]
        lis_txt_test_ques = transform_seg2uni(segmented_data=segmented_test_ques)
        embedding_test_ques = self.sent_bert_model.encode(lis_txt_test_ques, convert_to_tensor=True)
        lis_txt_corpus = transform_seg2uni(segmented_data=self.segmented_corpus)

        lis_predict_aid = []
        for i, ques_id in enumerate(tqdm(self.lis_test_idx)):
            top_n_aid = self.bm25_ranking.get_ranking(query_idx=ques_id, prefix='train_ques', top_n=top_n)
            top_n_article_txt = [lis_txt_corpus[aid] for aid in top_n_aid]
            embedding_article = self.sent_bert_model.encode(top_n_article_txt, convert_to_tensor=True)
            cosim_score = util.cos_sim(embedding_test_ques[i: i + 1], embedding_article)
            # top_n_sim_score = [cosim_score[ques_id][aid] for aid in top_n_aid]
            predict_aid = [aid for i, aid in enumerate(top_n_aid) if cosim_score[0][i] >= threshold]
            lis_predict_aid.append(predict_aid)

        sum_f2score = 0
        for i in range(len(self.lis_test_idx)):
            sum_f2score += calculate_f2i(lis_predict=lis_predict_aid[i], lis_ground_truth=lis_label[i])
        print('f2score: ', sum_f2score / len(self.lis_test_idx))


if __name__ == '__main__':
    sbert_model = SentenceTransformer(model_name_or_path=SENT_BERT_CHECKPOINT)
    evaluation = Evaluation(sent_bert_model=sbert_model,
                            test_idx_path=TEST_IDX,
                            segmented_corpus_path=SEGMENTED_LEGAL_CORPUS,
                            segmented_ques_path=SEGMENTED_DATA_QUESTION,
                            corpus_json_path=LEGAL_CORPUS_PATH,
                            ques_json_path=DATA_QUESTION_PATH)
    evaluation.start_evaluate(top_n=20)
