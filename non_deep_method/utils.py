from tqdm import tqdm

from gensim.models import KeyedVectors

vn_word_emb = KeyedVectors.load_word2vec_format('/Users/LongNH/Workspace/ZaloAIChallenge/large_files/vn.wiki.vi.vec',
                                                binary=False)


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
