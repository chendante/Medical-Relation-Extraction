import json
from transformers import BertTokenizer
import codecs


class DataProcess(object):
    def __init__(self, pretrained_path, raw_data_path, is_test=True):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.raw_data_path = raw_data_path
        self.is_test = is_test
        self.schemas = self.schemas_init()
        self.train_data = self.train_data_init()

    def schemas_init(self):
        schemas = []
        with codecs.open(self.raw_data_path + "train_data.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                schema = json.loads(line)
                schemas.append("_".join([schema["subject_type"], schema["predicate"], schema["object_type"]]))
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

    def __init__(self, pretrained_path, raw_data_path, is_test=True):
        super().__init__(pretrained_path, raw_data_path, is_test)
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
        return input_ids, attention_masks, labels

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
                encoded_dict = self.bert_tokenizer(sent, add_special_tokens=True)
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])

                label = '_'.join([instance['subject_type'], instance['predicate'], instance['object_type']['@value']])
                labels.append(self.schemas.index(label))
            else:
                print(spo)
                break
        return input_ids, attention_masks, labels


def main():
    subject_begin = "<s_b>"
    subject_end = "<s_e>"
    object_begin = "<o_b>"
    object_end = "<o_e>"
    special_tokens = {'additional_special_tokens': [subject_begin, subject_end, object_begin, object_end]}
    bert_tokenizer = BertTokenizer.from_pretrained("../pretrained_model/bert_wwn_ext/")
    bert_tokenizer.add_special_tokens(special_tokens)
    t = bert_tokenizer('<s_b>三尖瓣闭锁<s_e>@若患有感染性心内膜炎，细菌栓子亦可进入脑部。三尖瓣闭锁@所以，对于任何大于2岁的青紫型先心患儿，当出现头痛、呕吐、神经定位症状时，尚需考虑脑部疾病的存在。')
    # t = bert_tokenizer.cls_token
    print(t)


if __name__ == "__main__":
    main()
