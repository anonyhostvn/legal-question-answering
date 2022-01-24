from finetunning_bert.model_training import ModelTraining

if __name__ == '__main__':
    model_training = ModelTraining(train_idx_path='finetunning_bert/train_idx.json',
                                   test_idx_path='finetunning_bert/test_idx.json',
                                   cut_size=1, batch_size=1)
    model_training.start_training()
