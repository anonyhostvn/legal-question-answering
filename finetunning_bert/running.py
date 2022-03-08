from finetunning_bert.const import TRAIN_IDX_PATH, TEST_IDX_PATH
from finetunning_bert.model_training import ModelTraining
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-csz', '--cut_size', help='Size of dataset', default=None, type=int)
parser.add_argument('-bsz', '--batch_size', help='Size of batch', default=32, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    model_training = ModelTraining(train_idx_path=TRAIN_IDX_PATH,
                                   test_idx_path=TEST_IDX_PATH,
                                   cut_size=args.cut_size,
                                   batch_size=args.batch_size)
    model_training.start_training()
