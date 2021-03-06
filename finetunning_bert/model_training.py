from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_scheduler, AutoConfig

import torch

import numpy as np

from finetunning_bert.corpus_dataset import CorpusDataset

import json
from accelerate import Accelerator

from finetunning_bert.training_utilities import perform_epoch
from global_config import PRETRAIN_BERT_NAME, RAW_LEGAL_TEXT_CORPUS_PATH, LEGAL_BERT_MLM, PRETRAIN_BERT_TOKENIZER


class ModelTraining:
    def __init__(self, train_idx_path, test_idx_path,
                 cut_size=None, batch_size=32, mlm_prob=0.15, ):
        print('batch size : ', batch_size)
        self.save_folder = LEGAL_BERT_MLM
        print('save folder: ', self.save_folder)

        self.corpus_path = RAW_LEGAL_TEXT_CORPUS_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_BERT_TOKENIZER)
        self.bert_config = AutoConfig.from_pretrained(PRETRAIN_BERT_TOKENIZER)
        self.mlm_bert_model = AutoModelForMaskedLM.from_pretrained(PRETRAIN_BERT_NAME)
        self.mlm_prob = mlm_prob
        self.batch_size = batch_size

        with open(train_idx_path, 'r') as train_idx_file:
            lis_idx_train = json.load(train_idx_file)[:cut_size]
        with open(test_idx_path, 'r') as test_idx_file:
            lis_idx_test = json.load(test_idx_file)[:cut_size]

        print('Training samples: ', len(lis_idx_train))
        print('Testing samples: ', len(lis_idx_test))

        self.train_corpus_dataset = CorpusDataset(corpus_path=self.corpus_path, use_idx=lis_idx_train)
        self.test_corpus_dataset = CorpusDataset(corpus_path=self.corpus_path, use_idx=lis_idx_test)
        self.train_dataloader = DataLoader(dataset=self.train_corpus_dataset, batch_size=self.batch_size,
                                           collate_fn=self.collate_fn_dataloader)
        self.test_dataloader = DataLoader(dataset=self.test_corpus_dataset, batch_size=self.batch_size,
                                          collate_fn=self.collate_fn_dataloader)
        self.optimizer = AdamW(self.mlm_bert_model.parameters(), lr=5e-5)

    def random_masked_input(self, input_ids, attention_mask):
        """
        :param input_ids: MAX_SEQ_LENGTH
        :param attention_mask: MAX_SEQ_LENGTH
        :param mlm_prob: integer
        :return:
        """
        is_mask = np.random.binomial(1, self.mlm_prob, (len(input_ids),))
        label = torch.tensor(
            [input_ids[i] if is_mask[i] == 1 and attention_mask[i] == 1 else -100 for i in range(len(input_ids))])
        return label

    def custom_data_collator(self, encoding_result):
        """
        :param encoding_result: Bao g???m 3 keys (input_ids, attention_mask, token_type_ids)
        :return:
        """
        lis_input_ids = encoding_result['input_ids']
        lis_attention_mask = encoding_result['attention_mask']

        labels = []
        for sample_idx in range(len(lis_input_ids)):
            input_ids = lis_input_ids[sample_idx]
            attention_mask = lis_attention_mask[sample_idx]
            label = self.random_masked_input(input_ids=input_ids, attention_mask=attention_mask)
            labels.append(label)
        encoding_result['labels'] = torch.stack(tensors=labels, dim=0)
        return encoding_result

    def collate_fn_dataloader(self, batch):
        tokenizer_output = self.tokenizer(batch, padding='max_length', truncation='only_first',
                                          max_length=self.bert_config.max_position_embeddings, return_tensors='pt')
        data_tokenize_with_mlm = self.custom_data_collator(tokenizer_output)
        return data_tokenize_with_mlm

    def start_training(self):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        print('Training on ', device)
        self.mlm_bert_model = self.mlm_bert_model.to(device)

        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.mlm_bert_model, self.optimizer, self.train_dataloader, self.test_dataloader
        )

        num_train_epochs = 3
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        for epoch_id in range(5):
            perform_epoch(epoch_id=epoch_id, model=model, train_dataloader=train_dataloader,
                          eval_dataloader=eval_dataloader,
                          eval_dataset=self.test_corpus_dataset,
                          batch_size=self.batch_size, accelerator=accelerator,
                          optimizer=optimizer, lr_scheduler=lr_scheduler, device=device)

            model.save_pretrained(self.save_folder)


if __name__ == '__main__':
    model_training = ModelTraining(mlm_prob=0.15,
                                   train_idx_path='train_idx.json',
                                   test_idx_path='test_idx.json')
    model_training.start_training()
