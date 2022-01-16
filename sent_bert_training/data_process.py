from bm25_ranking.bm25_pre_ranking import bm25_ranking
from general_data.data_producer import DataProducer


class DataProcess:
    def __init__(self):
        self.bm25_ranking = bm25_ranking
        self.data_producer = DataProducer()

    def generate_se_sim_dataset(self):
        pass
