from sentence_transformers import SentenceTransformer, models, losses
from torch import nn
import torch

from global_config import SAVE_SENT_BERT_CHECKPOINT_PATH


class SentBertBuilder:
    def __init__(self, pretrain_model=None, pretrain_tokenize=None, pretrain_sent_bert=None):
        assert pretrain_sent_bert is not None or (pretrain_model is not None and pretrain_tokenize is not None)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if pretrain_sent_bert is None:
            word_embedding_model = models.Transformer(model_name_or_path=pretrain_model,
                                                      tokenizer_name_or_path=pretrain_tokenize, max_seq_length=256)
            word_embedding_model.to(self.device)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

            dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                       out_features=256, activation_function=nn.Tanh())

            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                             device=self.device)
        else:
            self.model = SentenceTransformer(model_name_or_path=pretrain_sent_bert)

        self.train_loss = losses.CosineSimilarityLoss(self.model)

    # Epoch: 100 % 4 / 4[6:23:56 < 00:00, 5759.10s / it]
    def start_training(self, train_dataloader):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        n_save_step = 1000
        print('Save checkpoint in ', SAVE_SENT_BERT_CHECKPOINT_PATH, 'after ', n_save_step, 'step')
        self.model = self.model.to(device)
        self.model.fit(train_objectives=[(train_dataloader, self.train_loss)], epochs=4, warmup_steps=100,
                       checkpoint_path=SAVE_SENT_BERT_CHECKPOINT_PATH, checkpoint_save_total_limit=1,
                       checkpoint_save_steps=n_save_step)
        self.model.save(SAVE_SENT_BERT_CHECKPOINT_PATH)
