import os

import numpy as np
from tqdm import tqdm
import fasttext

from non_deep_method.config import FT_EMBED_DIM, ESP, CACHE_DIR, FAST_TEXT_PRETRAINED_PATH

vn_word_emb = fasttext.load_model(FAST_TEXT_PRETRAINED_PATH)


def get_wavg_word_emb_with_cached(tfidf_score, vocab, cached_filename):
    cached_file_path = os.path.join(CACHE_DIR, cached_filename)
    # Load from cached
    if os.path.exists(cached_file_path):
        return np.load(cached_file_path)

    wavg_vect = np.zeros(shape=(FT_EMBED_DIM,), dtype=float)
    sum_w = 0
    for wid in tfidf_score.nonzero()[1]:
        wavg_vect += vn_word_emb.get_word_vector(vocab[wid]) * tfidf_score[0, wid]
        sum_w += tfidf_score[0, wid]
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


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


#
# list1 = ['dog', 'cat', 'cat', 'rat']
# list2 = ['dog', 'cat', 'mouse']
# jaccard_similarity(list1, list2)

if __name__ == '__main__':
    print('Start')
