from sentence_transformers import SentenceTransformer, models, evaluation, losses
from torch import nn
import torch


class SentBertBuilder:
    def __init__(self, pretrain_model, pretrain_tokenize):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        word_embedding_model = models.Transformer(model_name_or_path=pretrain_model,
                                                  tokenizer_name_or_path=pretrain_tokenize, max_seq_length=256)
        word_embedding_model.to(self.device)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256, activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)
        self.train_loss = losses.CosineSimilarityLoss(self.model)

    def start_training(self, train_dataloader):
        self.model.fit(train_objectives=[(train_dataloader, self.train_loss)], epochs=1, warmup_steps=100)
