import json
import os.path
import random

QUES_JSON = ''
DATA_QUES_SEGMENTED = '/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/train_ques_segmented.json'
TEST_QUES_SEGMENTED = '/Users/LongNH/Workspace/ZaloAIChallenge/segemented_data/test_ques_segmented.json'
TRAIN_QUES_PERCENT = 0.8


class GlobalSpliter:
    def __init__(self):
        self.train_idx_path = 'train_idx.json'
        self.test_idx_path = 'test_idx.json'
        self.lis_train_idx = None
        self.lis_test_idx = None

        if not os.path.exists(self.train_idx_path) or not os.path.exists(self.test_idx_path):
            print('Create from cached')
            with open(DATA_QUES_SEGMENTED, 'r') as f:
                data_ques_segmented = json.load(f)
            n_sample = len(data_ques_segmented)
            lis_idx = [i for i in range(n_sample)]
            random.shuffle(lis_idx)

            n_train_sample = int(n_sample * TRAIN_QUES_PERCENT)
            self.lis_train_idx = lis_idx[:n_train_sample]
            self.lis_test_idx = lis_idx[n_train_sample:]

            with open(self.train_idx_path, 'w') as f:
                json.dump(self.lis_train_idx, f)

            with open(self.test_idx_path, 'w') as f:
                json.dump(self.lis_test_idx, f)
        else:
            print('Load from cached')
            with open(self.train_idx_path, 'r') as f:
                self.lis_train_idx = json.load(f)
            with open(self.test_idx_path, 'r') as f:
                self.lis_test_idx = json.load(f)

        print('Train sample: ', len(self.lis_train_idx))
        print('Test sample: ', len(self.lis_test_idx))

    def get_train_idx(self):
        return self.lis_train_idx

    def get_test_idx(self):
        return self.lis_test_idx


global_spliter = GlobalSpliter()
