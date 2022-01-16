from torch.utils.data import Dataset


class SimSentDataset(Dataset):

    def __init__(self, lis_examples):
        self.lis_example = lis_examples

    def __len__(self):
        return len(self.lis_example)

    def __getitem__(self, index):
        return self.lis_example[index]
