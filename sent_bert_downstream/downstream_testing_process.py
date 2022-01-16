from global_config import SENT_BERT_CHECKPOINT
from sent_bert_downstream.data_process import DataProcess
from sent_bert_downstream.sent_bert_downstream import SentBertDownstream


class DownstreamTestingProcess:
    def __init__(self):
        self.data_process = DataProcess()
        self.sent_bert_model = SentBertDownstream(pretrain_sent_bert=SENT_BERT_CHECKPOINT)

    def cook_data(self):
        return self.data_process.generate_eval_se_sim_dataloader()

    def start_testing(self):
        test_data_dict = self.cook_data()
