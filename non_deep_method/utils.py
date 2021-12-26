import os

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

from non_deep_method.config import FT_EMBED_DIM, ESP, CACHE_DIR

vn_word_emb = KeyedVectors.load_word2vec_format('/Users/LongNH/Workspace/ZaloAIChallenge/large_files/vn.wiki.vi.vec',
                                                binary=False)


def get_wavg_word_emb_with_cached(tfidf_score, vocab, cached_filename):
    cached_file_path = os.path.join(CACHE_DIR, cached_filename)
    # Load from cached
    if os.path.exists(cached_file_path):
        return np.load(cached_file_path)

    wavg_vect = np.zeros(shape=(FT_EMBED_DIM,), dtype=float)
    sum_w = 0
    for wid, tfidf_score in enumerate(tfidf_score):
        if tfidf_score > 0:
            wavg_vect += vn_word_emb.get_vector(vocab[wid]) * tfidf_score
            sum_w += tfidf_score
    wavg_vect /= (sum_w + ESP)
    np.save(cached_file_path, wavg_vect)
    return wavg_vect


def transform_seg2bi(segmented_data):
    assert segmented_data is not None
    print('\n building bi corpus ... ')
    bi_corpus = [
        ' '.join([' '.join([f'{sent[i]}_{sent[i + 1]}' for i in range(len(sent) - 1)]) for sent in article])
        for article in tqdm(segmented_data)]
    return bi_corpus


def transform_seg2uni(segmented_data):
    print('\n building uni corpus ... ')
    uni_corpus = [' '.join([' '.join(sent) for sent in article]) for article in tqdm(segmented_data)]
    return uni_corpus


if __name__ == '__main__':
    print('Start')
