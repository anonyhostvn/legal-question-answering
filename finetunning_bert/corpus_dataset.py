from torch.utils.data import Dataset


class CorpusDataset(Dataset):
    def __init__(self, corpus_path, use_idx=None):
        with open(corpus_path, 'r') as f:
            self.corpus = f.readlines()
            if use_idx is not None:
                assert type(use_idx) == 'list'
                self.corpus = self.corpus[use_idx]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        return self.corpus[index]
