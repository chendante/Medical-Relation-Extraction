from preparation import prepare_data
import codecs
from transformers import *
import json
import jieba


class EntityBasedDataProcess(prepare_data.DataProcess):
    subject_begin = "<s_b>"
    subject_end = "<s_e>"
    object_begin = "<o_b>"
    object_end = "<o_e>"
    special_tokens = {'additional_special_tokens': [subject_begin, subject_end, object_begin, object_end]}

    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=300, is_test=True, test_data_path=None,
                 split_dic=None, voc_type_path=None):
        super().__init__(pretrained_path, raw_data_path, output_dir, max_len=max_len, is_test=is_test,
                         test_data_path=test_data_path)
        self.bert_tokenizer.add_special_tokens(self.special_tokens)
        jieba.load_userdict(split_dic)
        self.voc_type = None
        if voc_type_path is not None:
            with codecs.open(voc_type_path, encoding='utf-8') as f:
                self.voc_type = json.load(f, encoding='utf-8')
        self.blank_list = []

    def instance_process(self, instance):
        sent = instance['text']