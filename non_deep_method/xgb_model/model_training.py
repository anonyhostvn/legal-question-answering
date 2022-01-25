import json

from tqdm import tqdm

from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS, DATA_QUESTION_PATH, SEGMENTED_DATA_QUESTION, \
    TRAIN_IDX
from non_deep_method.corpus_builder.legal_corpus import LegalCorpus
from non_deep_method.corpus_builder.legal_question_corpus import LegalQuestionCorpus
from non_deep_method.data_builder.data_builder import DataBuilder
import numpy as np
import os

from non_deep_method.xgb_model.model_builder import ModelBuilder


class ModelTraining:
    def __init__(self, is_build=True):
        if is_build:
            self.legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH,
                                            corpus_segmented_path=SEGMENTED_LEGAL_CORPUS)
            self.train_ques_corpus = LegalQuestionCorpus(json_ques_path=DATA_QUESTION_PATH,
                                                         seg_ques_path=SEGMENTED_DATA_QUESTION)
            self.data_builder = DataBuilder(self.train_ques_corpus, self.legal_corpus)
        self.SAVE_X_PATH = os.path.join('large_files', 'train_x.npy')
        self.SAVE_Y_PATH = os.path.join('large_files', 'train_y.npy')

    def generate_idx_couple(self, lis_qidx, neg_ratio=5):
        lis_couple = []
        for qidx in tqdm(lis_qidx, desc='Building list of couple'):
            lis_relevant_aidx = self.data_builder.get_relevant_aidx_of_ques(qidx)
            lis_non_relevant_aidx = self.data_builder.get_non_relevant_aidx_of_ques_bm25(
                qidx, lis_relevant_aidx,
                n_elements=len(lis_relevant_aidx) * neg_ratio)
            for aidx in lis_relevant_aidx:
                lis_couple.append((qidx, aidx, 1))
            for aidx in lis_non_relevant_aidx:
                lis_couple.append((qidx, aidx, 0))
        return lis_couple

    def cook_training_data(self):
        with open(TRAIN_IDX, 'r') as f:
            lis_ques_idx = json.load(f)

        lis_couple = self.generate_idx_couple(lis_ques_idx)

        x = []
        y = []
        for qidx, aidx, label in tqdm(lis_couple, desc='Building feature vector'):
            x.append(self.data_builder.get_feature_vector(qidx, aidx))
            y.append(label)
        return np.array(x), np.array(y)

    def start_generate(self):
        x, y = self.cook_training_data()
        np.save(self.SAVE_X_PATH, x)
        np.save(self.SAVE_Y_PATH, y)

    def start_training(self):
        x = np.load(self.SAVE_X_PATH)
        y = np.load(self.SAVE_Y_PATH)
        model_builder = ModelBuilder()
        model_builder.train_k_fold(X=x, y=y)


if __name__ == '__main__':
    model_training = ModelTraining(is_build=False)
    # model_training.start_generate()
    model_training.start_training()
