import argparse

from finetunning_bert.model_training import ModelTraining

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("-model", "--model_name", help="Pretrained Model Name", required=True)

parser.add_argument("-bcorpus", "--bert_corpus", help="Bert Corpus Path", required=True)

parser.add_argument("-cut_size", "--cut_size", help="Size using for train", required=False)

parser.add_argument("-batch_size", "--batch_size", help="Batch size using for train", required=True)

parser.add_argument("-save_folder", "--save_folder", help="Folder for saved model weight", required=True)

args = vars(parser.parse_args())

if __name__ == '__main__':
    pretrained_model_name = args['model_name']
    bert_corpus_path = args['bert_corpus']
    cut_size = args['cut_size']
    cut_size = int(cut_size) if cut_size is not None else None
    batch_size = args['batch_size']
    batch_size = int(batch_size) if batch_size is not None else None
    save_folder = args['save_folder']
    model_training = ModelTraining(pretrain_name=pretrained_model_name, mlm_prob=0.15,
                                   tokenizer_name=pretrained_model_name, corpus_path=bert_corpus_path,
                                   train_idx_path='train_idx.json', test_idx_path='test_idx.json',
                                   cut_size=cut_size, batch_size=batch_size, save_folder=save_folder)
    model_training.start_training()
