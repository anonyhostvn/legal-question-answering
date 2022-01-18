import torch
from sentence_transformers import SentenceTransformer, losses, util

from global_config import SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH


class SentBertDownstream:
    def __init__(self, pretrain_sent_bert):
        assert pretrain_sent_bert is not None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name_or_path=pretrain_sent_bert)

        self.train_loss = losses.CosineSimilarityLoss(self.model)

    def start_training(self, train_dataloader):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_save_step = 1000
        print('Save checkpoint in ', SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH, 'after ', n_save_step, 'step using: ',
              device)
        self.model = self.model.to(device)
        self.model.fit(train_objectives=[(train_dataloader, self.train_loss)], epochs=2, warmup_steps=100,
                       checkpoint_path=SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH, checkpoint_save_total_limit=1,
                       checkpoint_save_steps=n_save_step)
        self.model.save(SENT_BERT_DOWNSTREAM_CHECKPOINT_PATH)

    def inferences(self, sent_1, lis_sent_2):
        self.model.eval()
        embedding_1 = self.model.encode([sent_1], convert_to_tensor=True)
        embedding_2 = self.model.encode(lis_sent_2, convert_to_tensor=True)
        cosine_score = util.cos_sim(embedding_1, embedding_2)
        cosine_result = []
        for i in range(len(lis_sent_2)):
            cosine_result.append(cosine_score[0, i])
        return cosine_result
