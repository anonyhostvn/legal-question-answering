from sentence_transformers import SentenceTransformer, models, evaluation
from torch import nn


class SentBertBuilder:
    def __init__(self, pretrain_model):
        word_embedding_model = models.Transformer(model_name_or_path=pretrain_model,
                                                  tokenizer_name_or_path=pretrain_model, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256, activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
