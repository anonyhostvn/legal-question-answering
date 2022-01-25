import json

from non_deep_method.corpus_builder.abstract_corpus import AbstractCorpus


class LegalQuestionCorpus(AbstractCorpus):
    def __init__(self, json_ques_path: str, seg_ques_path: str):
        with open(json_ques_path, 'r') as f:
            self.lis_json_ques = json.load(f).get('items')
        with open(seg_ques_path, 'r') as f:
            self.lis_seg_ques = json.load(f)
        super().__init__(self.lis_json_ques, self.lis_seg_ques)
