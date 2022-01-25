from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfMachine:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.tf_idf_corpus = self.vectorizer.fit_transform(corpus)

    def get_tfidf(self, idx):
        return self.tf_idf_corpus[idx]

    def transform_corpus(self, corpus):
        return self.vectorizer.transform(corpus)

    def get_transformed_root_corpus(self):
        return self.tf_idf_corpus
