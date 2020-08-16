import json
from transformers import BertTokenizer
import codecs
import numpy as np
import os
import thulac
from collections import defaultdict, Counter
from sklearn import preprocessing
import util
import re
import torch
from typing import List


class DataProcess(object):
    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=300, is_test=True, test_data_path=None):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.is_test = is_test
        self.output_dir = output_dir
        self.max_len = max_len
        self.schemas = self.schemas_init(raw_data_path)
        if test_data_path is None:
            self.train_data = self.train_data_init(raw_data_path, is_test)
        else:
            self.test_data = self.test_init(test_data_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def schemas_init(raw_data_path):
        schemas = []
        with codecs.open(raw_data_path + "53_schemas.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                schema = json.loads(line)
                schemas.append("_".join([schema['subject_type'], schema['predicate'], schema['object_type']]))
        return schemas

    @staticmethod
    def train_data_init(raw_data_path, is_test):
        train_data = []
        with codecs.open(raw_data_path + "train_data.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                train_data.append(json.loads(line))

        if not is_test:
            with codecs.open(raw_data_path + "val_data.json", encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if line == "":
                        break
                    train_data.append(json.loads(line))
        return train_data

    @staticmethod
    def test_init(test_data_path):
        test_data = []
        with codecs.open(test_data_path, encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                test_data.append(json.loads(line))
        return test_data

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
        labels = np.array(labels, dtype=np.float)
        np.save(self.output_dir + "input_ids.npy", input_ids)
        np.save(self.output_dir + "attention_masks.npy", attention_masks)
        np.save(self.output_dir + "labels.npy", labels)
        return input_ids, attention_masks, labels

    def instance_process(self, instance):
        raise Exception("Didn't implement instance_process")


class BeginEndDataProcess(DataProcess):
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
        self.lac = thulac.thulac(split_dic)
        self.voc_type = None
        if voc_type_path is not None:
            with codecs.open(voc_type_path, encoding='utf-8') as f:
                self.voc_type = json.load(f, encoding='utf-8')

    def instance_process(self, instance):
        # TODO
        # if len(token_list) > self.bert_tokenizer.model_max_length: pass  # BERT有输入上限
        input_ids = []
        attention_masks = []
        labels = []
        for spo in instance['spo_list']:
            if spo['object']['@value'] in instance['text'] and spo['subject'] in instance['text']:
                sent = instance['text']
                sent = sent.replace(spo['object']['@value'],
                                    self.object_begin + spo['object']['@value'] + self.object_end).replace(
                    spo['subject'], self.subject_begin + spo['subject'] + self.subject_end)
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

    def test_data_process(self):
        for t_d in self.test_data:
            sent = t_d['text']
            words = self.lac.cut(sent)
            medical_words = [w for w in words if w[0] in self.voc_type]
            if len(medical_words) < 2:
                # medical_words.extend([w for w in words if w[1] in ['n', 'np', 'ns', 'ni']])
                # if len(medical_words) <= 2:
                print(sent)
                print(medical_words)

    def test_instance_process(self):
        pass


class MultiLabelDataProcess(DataProcess):
    def instance_process(self, instance):
        # TODO
        # if len(token_list) > self.bert_tokenizer.model_max_length: pass  # BERT有输入上限
        input_ids = []
        attention_masks = []
        labels = []
        encoded_dict = self.bert_tokenizer(instance['text'], add_special_tokens=True, max_length=self.max_len,
                                           pad_to_max_length=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        label = np.array(
            [self.schemas.index('_'.join([spo['subject_type'], spo['predicate'], spo['object_type']['@value']])) for
             spo in instance["spo_list"]])
        one_hot = np.zeros(len(self.schemas), dtype=np.long)
        one_hot[label.ravel()] = 1
        labels.append(one_hot)

        return input_ids, attention_masks, labels

    def test_data_process(self):
        sent_list = [t_d['text'] for t_d in self.test_data]
        encode_dict = self.bert_tokenizer(sent_list, add_special_tokens=True, max_length=self.max_len,
                                          padding='max_length')
        input_ids = np.array(encode_dict['input_ids'], dtype=np.long)
        attention_masks = np.array(encode_dict['attention_mask'], dtype=np.long)
        # np.save(self.output_dir + "test_input_ids.npy", input_ids)
        # np.save(self.output_dir + "test_attention_masks.npy", attention_masks)
        return input_ids, attention_masks


class LabelDataProcess(DataProcess):
    sub_be_label = "B-SUB"
    obj_be_label = "B-OBJ"
    sub_fl_label = "I-SUB"
    obj_fl_label = "I-OBJ"
    nor_label = 'O'
    pad_label = 'P'
    token_labels = [pad_label, nor_label, sub_be_label, obj_be_label, sub_fl_label, obj_fl_label]

    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=512, is_test=True, test_data_path=None):
        super().__init__(pretrained_path, raw_data_path, output_dir, max_len=max_len, is_test=is_test,
                         test_data_path=test_data_path)
        # self.bert_tokenizer.add_special_tokens({'additional_special_tokens': self.schemas})

    def process(self):
        input_ids = []
        attention_masks = []
        labels = []
        token_type_ids = []
        for instance in self.train_data:
            i_input_ids, i_attention_masks, i_labels, i_token_ids = self.instance_process(instance)
            input_ids.extend(i_input_ids)
            attention_masks.extend(i_attention_masks)
            labels.extend(i_labels)
            token_type_ids.extend(i_token_ids)
        input_ids = np.array(input_ids, dtype=np.long)
        attention_masks = np.array(attention_masks, dtype=np.long)
        labels = np.array(labels, dtype=np.long)
        token_type_ids = np.array(token_type_ids, dtype=np.long)
        np.save(self.output_dir + "token_type_ids.npy", token_type_ids)
        np.save(self.output_dir + "input_ids.npy", input_ids)
        np.save(self.output_dir + "attention_masks.npy", attention_masks)
        np.save(self.output_dir + "labels.npy", labels)
        return input_ids, attention_masks, labels, token_type_ids

    def instance_process(self, instance):
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        token_type_ids_list = []
        for spo in instance['spo_list']:
            if spo['object']['@value'] in instance['text'] and spo['subject'] in instance['text']:
                schema = '_'.join([spo['subject_type'], spo['predicate'], spo['object_type']['@value']])
                # 英文在中文句子中出现时的 tokenize 方式有问题
                sent = self.flush_text(instance['text'])
                obj_text = self.flush_text(spo['object']['@value'])
                sub_text = self.flush_text(spo['subject'])
                sent_tokens = self.bert_tokenizer.tokenize(sent)
                sent_tokens = [self.bert_tokenizer.cls_token] + sent_tokens + [self.bert_tokenizer.sep_token]
                label_list = [self.nor_label] * len(sent_tokens)
                try:
                    if len(spo['object']['@value']) > len(spo['subject']):  # 防止覆盖情况（sub: AB obj: ABC)
                        self.tag_label_list(label_list, sent_tokens, sub_text, True)
                        self.tag_label_list(label_list, sent_tokens, obj_text, False)
                    else:
                        self.tag_label_list(label_list, sent_tokens, obj_text, False)
                        self.tag_label_list(label_list, sent_tokens, sub_text, True)
                except:
                    continue
                input_ids, attention_masks, token_type_ids, label_ids = self.convert_to_ids(label_list=label_list,
                                                                                            sent_tokens=sent_tokens,
                                                                                            schema_id=self.schemas.index(
                                                                                                schema))
                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_masks)
                labels_list.append(label_ids)
                token_type_ids_list.append(token_type_ids)
            else:
                print(spo)
                break
        return input_ids_list, attention_masks_list, labels_list, token_type_ids_list

    def test_data_process(self, r_label_data: List[torch.Tensor], threshold):
        train_to_sent = dict()
        train_index = 0
        input_ids_list = []
        attention_masks_list = []
        token_type_ids_list = []
        for i, (instance, r_d) in enumerate(zip(self.test_data, r_label_data)):
            pb = r_d.sigmoid()
            r_labels = [i for i, p in enumerate(pb) if p > threshold]
            for r_label in r_labels:
                train_to_sent[train_index] = (i, r_label)
                train_index += 1
                # 英文在中文句子中出现时的 tokenize 方式有问题
                sent = self.flush_text(instance['text'])
                sent_tokens = self.bert_tokenizer.tokenize(sent)
                sent_tokens = [self.bert_tokenizer.cls_token] + sent_tokens + [self.bert_tokenizer.sep_token]
                input_ids, attention_masks, token_type_ids = self.convert_to_ids(sent_tokens=sent_tokens,
                                                                                 schema_id=r_label)
                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_masks)
                token_type_ids_list.append(token_type_ids)
        return input_ids_list, attention_masks_list, token_type_ids_list, train_to_sent

    def tag_label_list(self, label_list, sent_tokens, label_text, sub_or_obj):
        label_tokens = self.bert_tokenizer.tokenize(label_text)
        label_indexes = util.index_of_sentence(label_tokens, sent_tokens)
        if not label_indexes:
            label_tokens[0] = "##" + label_tokens[0]
            label_indexes = util.index_of_sentence(label_tokens, sent_tokens)
            if not label_indexes:
                print(label_tokens)
                print(sent_tokens)
                print(label_text)
                # return []
                raise Exception("label didn't show in text")
        for idx in label_indexes:
            label_list[idx] = self.sub_be_label if sub_or_obj else self.obj_be_label
            label_list[idx + 1:idx + len(label_tokens)] = [self.sub_fl_label if sub_or_obj else self.obj_fl_label] * (
                    len(label_tokens) - 1)
        return label_list

    def convert_to_ids(self, sent_tokens, schema_id, label_list=None):
        token_type_ids = [0 for _ in sent_tokens]

        # convert to id
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(sent_tokens)

        # add question part
        input_ids.extend([schema_id + 2] * (len(sent_tokens) - 2))
        input_ids = input_ids[:self.max_len - 1]
        input_ids.append(self.bert_tokenizer.sep_token_id)
        no_pad_len = len(input_ids)
        token_type_ids.extend([1 for _ in range(no_pad_len - len(token_type_ids))])
        attention_masks = [1] * no_pad_len

        # pad
        input_ids.extend([0] * (self.max_len - len(input_ids)))
        attention_masks.extend([0] * (self.max_len - len(attention_masks)))
        token_type_ids.extend([0] * (self.max_len - len(token_type_ids)))

        assert len(input_ids) == self.max_len
        assert len(attention_masks) == self.max_len
        assert len(token_type_ids) == self.max_len
        if label_list is not None:
            label_ids = [self.token_labels.index(l) for l in label_list]
            label_ids.extend([self.token_labels.index(self.nor_label)] * (no_pad_len - len(label_ids)))
            label_ids.extend([0] * (self.max_len - len(label_ids)))
            assert len(label_ids) == self.max_len
            return input_ids, attention_masks, token_type_ids, label_ids
        return input_ids, attention_masks, token_type_ids

    @staticmethod
    def flush_text(text):
        return re.sub("([a-zA-Z0-9]+\\s)*[a-zA-Z0-9]+", lambda x: " " + x.group() + " ", text)


def create_dictionary(diction_path, json_path):
    dictionary = defaultdict(Counter)
    train_data = DataProcess("../pretrained_model/bert_wwm/", "../raw_data/", "./processed_data/", is_test=False)
    for td in train_data.train_data:
        sent = td['text']
        index = sent.find("@")
        if index != -1:
            title = sent[0:index]
            if "。" not in title:
                dictionary[title]["疾病"] += 1
        for spo in td['spo_list']:
            dictionary[spo['object']['@value']][spo['object_type']['@value']] += 1
            dictionary[spo['subject']][spo['subject_type']] += 1
    counter = Counter()
    for k, v in dictionary.items():
        counter[k] = sum(v.values())
    with codecs.open(diction_path, 'w+', encoding='utf-8') as f:
        for k, v in counter.most_common():
            f.write(k + "\t" + str(v) + "\n")
    with codecs.open(json_path, 'w+', encoding='utf-8') as f:
        json.dump(dictionary, f)


def main():
    data_processor = LabelDataProcess("../pretrained_model/bert_wwm/", "../raw_data/", "./processed_data/token_label/")
    data_processor.process()
    # s = data_processor.bert_tokenizer(["人人为我", "我为人人"], max_length=10, padding="max_length")


if __name__ == "__main__":
    main()
    print(LabelDataProcess.obj_be_label)
