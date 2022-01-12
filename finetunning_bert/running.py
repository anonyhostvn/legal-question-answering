import argparse

from finetunning_bert.model_training import ModelTraining

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-model", "--model_name", help="Pretrained Model Name", required=True)

parser.add_argument("-bcorpus", "--bert_corpus", help="Bert Corpus Path", required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    pretrained_model_name = args['model_name']
    bert_corpus_path = args['bert_corpus']
    model_training = ModelTraining(pretrain_name=pretrained_model_name, mlm_prob=0.15,
                                   tokenizer_name=pretrained_model_name, corpus_path=bert_corpus_path,
                                   train_idx_path='train_idx.json',
                                   test_idx_path='test_idx.json')
    model_training.start_training()
