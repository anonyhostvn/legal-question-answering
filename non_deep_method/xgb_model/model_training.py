import json

from tqdm import tqdm

from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS, DATA_QUESTION_PATH, SEGMENTED_DATA_QUESTION, \
    TRAIN_IDX
from non_deep_method.corpus_builder.legal_corpus import LegalCorpus
from non_deep_method.corpus_builder.legal_question_corpus import LegalQuestionCorpus
from non_deep_method.data_builder.data_builder import DataBuilder


class ModelTraining:
    def __init__(self):
        self.legal_corpus = LegalCorpus(corpus_json_path=LEGAL_CORPUS_PATH,
                                        corpus_segmented_path=SEGMENTED_LEGAL_CORPUS)
        self.train_ques_corpus = LegalQuestionCorpus(json_ques_path=DATA_QUESTION_PATH,
                                                     seg_ques_path=SEGMENTED_DATA_QUESTION)
        self.data_builder = DataBuilder(self.train_ques_corpus, self.legal_corpus)

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
        return x, y

    def start_training(self):
        x, y = self.cook_training_data()
        print('test')


if __name__ == '__main__':
    model_training = ModelTraining()
    model_training.start_training()
