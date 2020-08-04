import json
from transformers import BertTokenizer
import codecs
import numpy as np
import os


class DataProcess(object):
    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=300, is_test=True):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.raw_data_path = raw_data_path
        self.is_test = is_test
        self.schemas = self.schemas_init()
        self.train_data = self.train_data_init()
        self.output_dir = output_dir
        self.max_len = max_len
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def schemas_init(self):
        schemas = []
        with codecs.open(self.raw_data_path + "53_schemas.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                schema = json.loads(line)
                schemas.append("_".join([schema['subject_type'], schema['predicate'], schema['object_type']]))
        return schemas

    def train_data_init(self):
        train_data = []
        with codecs.open(self.raw_data_path + "train_data.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                train_data.append(json.loads(line))

        if not self.is_test:
            with codecs.open(self.raw_data_path + "val_data.json", encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    train_data.append(json.loads(line))
        return train_data

    def process(self):
        pass

    def instance_process(self, instance):
        pass


class BeginEndDataProcess(DataProcess):
    subject_begin = "<s_b>"
    subject_end = "<s_e>"
    object_begin = "<o_b>"
    object_end = "<o_e>"
    special_tokens = {'additional_special_tokens': [subject_begin, subject_end, object_begin, object_end]}

    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=300, is_test=True):
        super().__init__(pretrained_path, raw_data_path, output_dir, max_len, is_test)
        self.bert_tokenizer.add_special_tokens(self.special_tokens)

    def process(self):
        input_ids = []
        attention_masks = []
        labels = []
        for instance in self.train_data:
            i_input_ids, i_attention_masks, i_labels = self.instance_process(instance)
            input_ids.extend(i_input_ids)
            attention_masks.extend(i_attention_masks)
            labels.extend(i_labels)
        input_ids = np.array(input_ids, dtype=np.long)
        attention_masks = np.array(attention_masks, dtype=np.long)
        labels = np.array(labels)
        np.save(self.output_dir + "input_ids.npy", input_ids)
        np.save(self.output_dir + "attention_masks.npy", attention_masks)
        np.save(self.output_dir + "labels.npy", labels)

    def instance_process(self, instance):
        # TODO
        # if len(token_list) > self.bert_tokenizer.model_max_length: pass  # BERT有输入上限
        input_ids = []
        attention_masks = []
        labels = []
        for spo in instance['spo_list']:
            if spo['object']['@value'] in instance['text'] and spo['subject'] in instance['text']:
                sent = instance['text']
                sent.replace(spo['object']['@value'], self.object_begin + spo['object']['@value'] + self.object_end)
                sent.replace(spo['subject'], self.subject_begin + spo['subject'] + self.subject_end)
                encoded_dict = self.bert_tokenizer(sent, add_special_tokens=True, pad_to_max_length=True,
                                                   max_length=self.max_len)
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
                label = '_'.join([spo['subject_type'], spo['predicate'], spo['object_type']['@value']])
                labels.append(self.schemas.index(label))
            else:
                print(spo)
                break
        return input_ids, attention_masks, labels


def main():
    data_processor = BeginEndDataProcess("../pretrained_model/bert_wwm/", "../raw_data/", "./processed_data/")
    data_processor.process()
    # subject_begin = "<s_b>"
    # subject_end = "<s_e>"
    # object_begin = "<o_b>"
    # object_end = "<o_e>"
    # special_tokens = {'additional_special_tokens': [subject_begin, subject_end, object_begin, object_end]}
    # bert_tokenizer = BertTokenizer.from_pretrained("../pretrained_model/bert_wwm/")
    # bert_tokenizer.add_special_tokens(special_tokens)
    # sent = '<s_b>三尖瓣闭锁<s_e>@若患有感染性心内膜炎，细菌栓子亦可进入脑部。三尖瓣闭锁@定位症状时，尚需考虑脑部疾病的存在。'
    # t = bert_tokenizer(sent, max_length=300, pad_to_max_length=True)
    # # t = bert_tokenizer.cls_token
    # print(t, len(t['input_ids']))


if __name__ == "__main__":
    main()
