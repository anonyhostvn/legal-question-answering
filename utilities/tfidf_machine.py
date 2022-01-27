import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utilities.text_preprocessing import mono_text_preprocessing


class TfIdfMachine:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.pre_processing = mono_text_preprocessing
        self.corpus = [mono_text_preprocessing.process(doc) for doc in tqdm(corpus, desc='Preprocess corpus')]
        self.tf_idf_corpus = self.vectorizer.fit_transform(self.corpus)

    def get_tfidf(self, idx):
        return self.tf_idf_corpus[idx]

    def transform_corpus(self, corpus):
        return self.vectorizer.transform(corpus)

    def get_transformed_root_corpus(self):
        return self.tf_idf_corpus
