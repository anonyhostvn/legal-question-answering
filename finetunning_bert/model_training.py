from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer

from finetunning_bert.const import PRETRAINED_MODEL_NAME, BERT_CORPUS_PATH
import torch

import numpy as np

from finetunning_bert.corpus_dataset import CorpusDataset


class ModelTraining:
    def __init__(self, pretrain_name, mlm_prob, tokenizer_name, corpus_path, batch_size=10):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.mlm_bert_model = AutoModelForMaskedLM.from_pretrained(pretrain_name)
        self.mlm_prob = mlm_prob
        self.corpus_dataset = CorpusDataset(corpus_path=corpus_path)
        self.train_dataloader = DataLoader(dataset=self.corpus_dataset, batch_size=batch_size,
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
        return torch.tensor(label)

    def custom_data_collator(self, encoding_result):
        """
        :param encoding_result: Bao gồm 3 keys (input_ids, attention_mask, token_type_ids)
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
                                          return_tensors='pt')
        data_tokenize_with_mlm = self.custom_data_collator(tokenizer_output)
        return data_tokenize_with_mlm

    def start_training(self):
        self.mlm_bert_model.train()
        for batch in self.train_dataloader:
            print(batch)


if __name__ == '__main__':
    model_training = ModelTraining(pretrain_name=PRETRAINED_MODEL_NAME, mlm_prob=0.15,
                                   tokenizer_name=PRETRAINED_MODEL_NAME, corpus_path=BERT_CORPUS_PATH)
    model_training.start_training()
