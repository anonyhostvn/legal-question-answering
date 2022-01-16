import torch
from sentence_transformers import SentenceTransformer, losses

from global_config import SAVE_SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH


class SentBertDownstream:
    def __init__(self, pretrain_sent_bert):
        assert pretrain_sent_bert is not None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name_or_path=pretrain_sent_bert)

        self.train_loss = losses.CosineSimilarityLoss(self.model)

    def start_training(self, train_dataloader):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_save_step = 1000
        print('Save checkpoint in ', SAVE_SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH, 'after ', n_save_step, 'step using: ',
              device)
        self.model = self.model.to(device)
        self.model.fit(train_objectives=[(train_dataloader, self.train_loss)], epochs=2, warmup_steps=100,
                       checkpoint_path=SAVE_SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH, checkpoint_save_total_limit=1,
                       checkpoint_save_steps=n_save_step)
        self.model.save(SAVE_SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH)
