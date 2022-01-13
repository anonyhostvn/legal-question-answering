import argparse

from sentence_transformers import SentenceTransformer, util
import json

from bm25_ranking.bm25_pre_ranking import bm25_ranking
from non_deep_method.evaluation.eval_utils import calculate_f2i
from sent_bert_finetune.sent_bert_builder import SentBertBuilder
from utilities.utils import transform_seg2uni


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

        with open(ques_json_path, 'r') as f:
            self.lis_ques = json.load(f).get('items')
        with open(corpus_json_path, 'r') as f:
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
        # embedding_corpus_ques = self.sent_bert_model.encode(lis_txt_corpus, convert_to_tensor=True)

        # cosim_score = util.cos_sim(embedding_test_ques, embedding_corpus_ques)
        lis_predict_aid = []

        for i, ques_id in enumerate(self.lis_test_idx):
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


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-model", "--model_name", help="Pretrained Model Name", default='vinai/phobert-base')

parser.add_argument('-test_idx_path', '--test_idx_path', help='Test idx path',
                    default='/Users/LongNH/Workspace/ZaloAIChallenge/data_spliter/test_idx.json')

parser.add_argument('-seg_cor_path', '--segmented_corpus_path', help='Segmented corpus path',
                    default='/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/segmented_corpus.json')

parser.add_argument('-seg_ques_path', '--segmented_ques_path', help='Segmented question path',
                    default='/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/train_ques_segmented.json')

parser.add_argument('-cor_json_path', '--corpus_json_path', help='Corpus json path',
                    default='/Users/LongNH/Workspace/ZaloAIChallenge/zac2021-ltr-data/legal_corpus.json')

parser.add_argument('-ques_json_path', '--question_json_path', help='Question json path',
                    default='/Users/LongNH/Workspace/ZaloAIChallenge/zac2021-ltr-data/train_question_answer.json')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    print(args)

    sbert_model = SentBertBuilder(pretrain_model=args['model_name'])
    evaluation = Evaluation(sent_bert_model=sbert_model.model,
                            test_idx_path=args['test_idx_path'],
                            segmented_corpus_path=args['segmented_corpus_path'],
                            segmented_ques_path=args['segmented_ques_path'],
                            corpus_json_path=args['corpus_json_path'],
                            ques_json_path=args['question_json_path'])
    evaluation.start_evaluate(top_n=20)
