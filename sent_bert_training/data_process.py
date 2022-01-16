from bm25_ranking.bm25_pre_ranking import bm25_ranking
from general_data.data_producer import DataProducer
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from utilities.sim_sent_dataset import SimSentDataset


class DataProcess:
    def __init__(self):
        self.bm25_ranking = bm25_ranking
        self.data_producer = DataProducer()
        self.top_n = 50
        self.batch_size = 16

    def generate_se_sim_dataset(self):
        lis_example = []
        for ques_idx in self.data_producer.lis_train_idx:
            segmented_ques = self.data_producer.get_segmented_ques(ques_idx)
            txt_ques = ' '.join([tok for sent in segmented_ques for tok in sent])

            gtruth_relevance_aidx = self.data_producer.get_ground_truth_relevance_article_ques(ques_idx)
            bm25_ranking_aidx = self.bm25_ranking.get_ranking(query_idx=ques_idx, prefix='train_ques', top_n=self.top_n)

            for aidx in gtruth_relevance_aidx:
                segmented_article = self.data_producer.get_segmented_legal_article(aidx)
                txt_article = ' '.join([tok for sent in segmented_article for tok in sent])
                lis_example.append(InputExample(texts=[txt_ques, txt_article], label=1))

            for aidx in bm25_ranking_aidx:
                if aidx not in gtruth_relevance_aidx:
                    segmented_article = self.data_producer.get_segmented_legal_article(aidx)
                    txt_article = ' '.join([tok for sent in segmented_article for tok in sent])
                    lis_example.append(InputExample(texts=[txt_ques, txt_article], label=0))

        return lis_example

    def generate_se_sim_dataloader(self):
        lis_example = self.generate_se_sim_dataset()

        def custom_collate_fn(batch_data):
            return batch_data

        sim_sent_dataset = SimSentDataset(lis_examples=lis_example)
        return DataLoader(sim_sent_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=custom_collate_fn)

    def generate_eval_se_sim_dataset(self):
        eval_se_sim_dataset = []
        for ques_idx in self.data_producer.lis_test_idx:
            segmented_ques = self.data_producer.get_segmented_ques(ques_idx)
            txt_ques = ' '.join([tok for sent in segmented_ques for tok in sent])

            gtruth_relevance_aidx = self.data_producer.get_ground_truth_relevance_article_ques(ques_idx)

            bm25_ranking_aidx = self.bm25_ranking.get_ranking(query_idx=ques_idx, prefix='train_ques', top_n=self.top_n)

            feed_forward_data = []

            for aidx in bm25_ranking_aidx:
                segmented_article = self.data_producer.get_segmented_legal_article(aidx)
                txt_article = ' '.join([tok for sent in segmented_article for tok in sent])
                feed_forward_data.append(InputExample(texts=[txt_ques, txt_article], label=0))

            eval_se_sim_dataset.append({
                'gtruth': gtruth_relevance_aidx,
                'bm25_ranking': bm25_ranking_aidx,
                'feed_forward_data': feed_forward_data
            })

            return eval_se_sim_dataset

    def generate_eval_se_sim_dataloader(self):
        lis_eval_se_sim = self.generate_eval_se_sim_dataset()

        def custom_collate_fn(batch_data):
            return batch_data

        for eval_id in range(len(lis_eval_se_sim)):
            lis_feed_forward = lis_eval_se_sim[eval_id].get('feed_forward_data')
            sim_sent_dataset = SimSentDataset(lis_examples=lis_feed_forward)
            sim_sent_dataloader = DataLoader(sim_sent_dataset, shuffle=False, batch_size=self.batch_size,
                                             collate_fn=custom_collate_fn)
            lis_eval_se_sim[eval_id]['feed_forward_dataloader'] = sim_sent_dataloader
            lis_eval_se_sim[eval_id].pop('feed_forward_data', None)

        return lis_eval_se_sim
