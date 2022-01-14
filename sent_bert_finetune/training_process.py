from global_config import SEGMENTED_LEGAL_CORPUS, SEGMENTED_DATA_QUESTION, TEST_IDX, SEGMENTED_TITLE_PATH
from sent_bert_finetune.data_preprocess import DataPreprocess
from sent_bert_finetune.sent_bert_builder import SentBertBuilder


class TrainingProcess:
    def __init__(self):
        pass

    @staticmethod
    def start_training():
        data_preprocess = DataPreprocess(segmented_corpus_path=SEGMENTED_LEGAL_CORPUS,
                                         segmented_ques_path=SEGMENTED_DATA_QUESTION,
                                         test_idx_path=TEST_IDX,
                                         segmented_title_path=SEGMENTED_TITLE_PATH)
        train_dataloader = data_preprocess.generate_dataloader_for_semantic_sim()

        sent_bert_builder = SentBertBuilder(pretrain_model='vinai/phobert-base', pretrain_tokenize='vinai/phobert-base')
        sent_bert_builder.start_training(train_dataloader)


if __name__ == '__main__':
    TrainingProcess.start_training()
