import json
import os

import numpy as np
from tqdm import tqdm

from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS, DATA_QUESTION_PATH, SEGMENTED_DATA_QUESTION, \
    TEST_IDX
from non_deep_method.corpus_builder.legal_corpus import LegalCorpus
from non_deep_method.corpus_builder.legal_question_corpus import LegalQuestionCorpus
from non_deep_method.data_builder.data_builder import DataBuilder
import pickle

from non_deep_method.evaluation.eval_utils import calculate_f2i, calculate_f2i_spec
from non_deep_method.xgb_model.model_builder import ModelBuilder


class ModelEvaluation:
    def __init__(self, is_build=True):
        if is_build:
            self.legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH,
                                            corpus_segmented_path=SEGMENTED_LEGAL_CORPUS)
            self.train_ques_corpus = LegalQuestionCorpus(json_ques_path=DATA_QUESTION_PATH,
                                                         seg_ques_path=SEGMENTED_DATA_QUESTION)
            self.data_builder = DataBuilder(self.train_ques_corpus, self.legal_corpus)

        self.SAVE_X_PATH = os.path.join('large_files', 'train_x.npy')
        self.SAVE_Y_PATH = os.path.join('large_files', 'train_y.npy')
        self.SAVE_LIS_MODEL_PATH = os.path.join('non_deep_method', 'cached', 'lis_xgb.pkl')
        self.SAVE_TEST = os.path.join('non_deep_method', 'cached', 'lis_test_result.pkl')

    def start_choosing_threshold(self):
        assert os.path.exists(self.SAVE_TEST), 'File result is not exist'
        test_result = pickle.load(open(self.SAVE_TEST, 'rb'))
        for threshold in np.arange(0, 1, 0.01):
            sum_f2score = 0
            sum_recall = 0
            sum_precision = 0
            for x, lis_bm25, py, gtruth_aidx in test_result:
                predict_aidx = []
                for i, prob in enumerate(py):
                    if prob >= threshold:
                        predict_aidx.append(lis_bm25[i])
                f2score, precision, recall = calculate_f2i_spec(lis_predict=predict_aidx, lis_ground_truth=gtruth_aidx)
                sum_f2score += f2score
                sum_recall += recall
                sum_precision += precision

                # sum_f2score += calculate_f2i(lis_predict=predict_aidx, lis_ground_truth=gtruth_aidx)
            print('Threshold = ', threshold, 'f2score = ', sum_f2score / len(test_result),
                  'recall = ', sum_recall / len(test_result),
                  'precision = ', sum_precision / len(test_result))

    def start_evaluate(self):
        lis_model = pickle.load(open(self.SAVE_LIS_MODEL_PATH, 'rb'))
        model_builder = ModelBuilder(lis_clf=lis_model)
        with open(TEST_IDX, 'r') as f:
            lis_ques_idx = json.load(f)

        test_result = []
        for qidx in tqdm(lis_ques_idx):
            lis_bm25_ranking = self.data_builder.get_bm25_ranking(qidx=qidx, top_n=50)
            x = [self.data_builder.get_feature_vector(qidx, aidx) for aidx in lis_bm25_ranking]
            py = model_builder.predict_prob(x)
            g_truth = self.data_builder.get_relevant_aidx_of_ques(qidx)
            test_result.append((list(x), list(lis_bm25_ranking), list(py), list(g_truth)))

        pickle.dump(test_result, open(self.SAVE_TEST, 'wb'))


if __name__ == '__main__':
    model_eval = ModelEvaluation(is_build=False)
    # model_eval.start_evaluate()
    model_eval.start_choosing_threshold()
