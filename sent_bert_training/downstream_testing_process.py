from global_config import SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH
from non_deep_method.evaluation.eval_utils import calculate_f2i
from sent_bert_training.data_process import DataProcess
from sent_bert_training.sent_bert_downstream import SentBertDownstream


class DownstreamTestingProcess:
    def __init__(self):
        self.data_process = DataProcess()
        self.sent_bert_model = SentBertDownstream(pretrain_sent_bert=SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH)

    def cook_data(self):
        return self.data_process.generate_eval_se_sim_dataset()

    def start_pure_testing(self, threshold=0.6):
        lis_test_data_dict = self.cook_data()
        sum_f2score = 0
        for test_data_dict in lis_test_data_dict:
            txt_ques = test_data_dict.get('question')
            lis_txt_article = test_data_dict.get('lis_candidate_article')
            lis_cosine_score = self.sent_bert_model.inferences(txt_ques, lis_txt_article)

            lis_predict_aidx = []
            for i, aidx in enumerate(test_data_dict.get('bm25_ranking')):
                if lis_cosine_score[i] >= threshold:
                    lis_predict_aidx.append(aidx)

            sum_f2score += calculate_f2i(lis_ground_truth=test_data_dict.get('gtruth'), lis_predict=lis_predict_aidx)

        print('f2score: ', sum_f2score / len(lis_test_data_dict))


if __name__ == '__main__':
    downstream_test_process = DownstreamTestingProcess()
    downstream_test_process.start_pure_testing()
