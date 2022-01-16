from global_config import SENT_BERT_CHECKPOINT
from sent_bert_training.data_process import DataProcess
from sent_bert_training.sent_bert_downstream import SentBertDownstream


class DownstreamTrainingProcess:
    def __init__(self):
        self.data_process = DataProcess()
        self.sent_bert_builder = SentBertDownstream(pretrain_sent_bert=SENT_BERT_CHECKPOINT)

    def cook_data(self):
        return self.data_process.generate_se_sim_dataloader()

    def start_training(self):
        train_data_loader = self.cook_data()
        self.sent_bert_builder.start_training(train_data_loader)


if __name__ == '__main__':
    training_process = DownstreamTrainingProcess()
    training_process.start_training()
