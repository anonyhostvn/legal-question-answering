from global_config import LEGAL_CORPUS_PATH, SEGMENTED_LEGAL_CORPUS, DATA_QUESTION_PATH, SEGMENTED_DATA_QUESTION
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

    def cook_data(self, lis_qidx):
        for qidx in lis_qidx:
            pass

    def start_training(self):
        pass
