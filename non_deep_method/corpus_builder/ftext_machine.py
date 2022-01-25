import fasttext

from non_deep_method.config import FAST_TEXT_PRETRAINED_PATH, FT_EMBED_DIM, ESP
import numpy as np


class FtextMachine:
    def __init__(self):
        self.vn_word_emb = fasttext.load_model(FAST_TEXT_PRETRAINED_PATH)

    def get_wavg_word_emb(self, tfidf_score, vocab):
        wavg_vect = np.zeros(shape=(FT_EMBED_DIM,), dtype=float)
        sum_w = 0
        for wid in tfidf_score.nonzero()[1]:
            wavg_vect += self.vn_word_emb.get_word_vector(vocab[wid]) * tfidf_score[0, wid]
            sum_w += tfidf_score[0, wid]
        wavg_vect /= (sum_w + ESP)
        return wavg_vect


ftext_machine = FtextMachine()
