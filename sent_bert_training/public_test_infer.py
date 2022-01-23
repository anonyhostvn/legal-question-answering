from bm25_ranking.bm25_pre_ranking import bm25_ranking
from global_config import SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH
from sent_bert_training.sent_bert_downstream import SentBertDownstream


class PublicTestInfer:
    def __init__(self):
        self.model = SentBertDownstream(pretrain_sent_bert=SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH)
        self.bm25 = bm25_ranking
