from preparation import prepare_data
import codecs
from transformers import *
import json
import jieba
from collections import defaultdict
import util
import torch
from typing import List, Set, Dict


class DiseaseBasedDataProcess(prepare_data.DataProcess):
    subject_begin = "<s_b>"
    subject_end = "<s_e>"
    special_tokens = {'additional_special_tokens': [subject_begin, subject_end]}
    be_label = "B"
    fl_label = "I"
    nor_label = "O"
    pad_label = 'P'
    token_labels = [pad_label, nor_label, be_label, fl_label]

    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=512, is_test=True, test_data_path=None,
                 split_dic=None, voc_type_path=None):
        super().__init__(pretrained_path, raw_data_path, output_dir, max_len=max_len, is_test=is_test,
                         test_data_path=test_data_path)
        self.bert_tokenizer.add_special_tokens(self.special_tokens)
        self.cut_tokenizer = None
        if split_dic is not None:
            self.cut_tokenizer = jieba.Tokenizer()
            self.cut_tokenizer.load_userdict(split_dic)
        self.voc_type = None
        self.disease_list = set()
        if voc_type_path is not None:
            with codecs.open(voc_type_path, encoding='utf-8') as f:
                self.voc_type = json.load(f, encoding='utf-8')
            for v, types in self.voc_type.items():
                if "疾病" in types and len(v) >= 1:
                    self.disease_list.add(v)

    @staticmethod
    def schemas_init(raw_data_path):
        schemas = []
        with codecs.open(raw_data_path + "53_schemas.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                schema = json.loads(line)
                if schema['subject_type'] == "疾病":
                    schemas.append("_".join([schema['subject_type'], schema['predicate'], schema['object_type']]))
        return schemas

    def instance_process(self, instance):
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        sent = instance['text']
        # sent = util.flush_text(instance['text'])
        disease_dict = defaultdict(list)
        for spo in instance['spo_list']:
            if spo['subject_type'] == "疾病":
                disease_dict[spo['subject']].append(spo["object"]["@value"])

        for disease, obj_list in disease_dict.items():
            obj_list.sort(key=lambda x: len(x), reverse=True)
            disease_sent = sent.replace(disease, self.subject_begin + disease + self.subject_end)
            sent_tokens = self.bert_tokenizer.tokenize(disease_sent)
            sent_tokens = [self.bert_tokenizer.cls_token] + sent_tokens + [self.bert_tokenizer.sep_token]
            label_list = [self.nor_label] * len(sent_tokens)
            for obj_text in obj_list:
                try:
                    self.tag_label_list(label_list, sent_tokens, obj_text)
                except:
                    continue
            input_ids, attention_masks, label_ids = self.convert_to_ids(label_list=label_list, sent_tokens=sent_tokens)
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_masks)
            labels_list.append(label_ids)
        return input_ids_list, attention_masks_list, labels_list

    def tag_label_list(self, label_list, sent_tokens, label_text):
        label_tokens = self.bert_tokenizer.tokenize(label_text)
        label_indexes = util.index_of_sentence(label_tokens, sent_tokens)
        if not label_indexes:
            label_tokens[0] = "##" + label_tokens[0]
            label_indexes = util.index_of_sentence(label_tokens, sent_tokens)
            if not label_indexes:
                # print(label_tokens)
                # print(sent_tokens)
                # print(label_text)
                raise Exception("label didn't show in text")
        for idx in label_indexes:
            label_list[idx] = self.be_label
            label_list[idx + 1:idx + len(label_tokens)] = [self.fl_label] * (len(label_tokens) - 1)
        return label_list

    def test_data_process(self):
        train_to_sent = dict()
        train_index = 0
        input_ids_list = []
        attention_masks_list = []
        for i, instance in enumerate(self.test_data):
            sent = instance['text']
            diseases = self.get_disease(sent)
            for disease in diseases:
                train_to_sent[train_index] = (i, disease)
                train_index += 1
                disease_sent = sent.replace(disease, self.subject_begin + disease + self.subject_end)
                # disease_sent = util.flush_text(disease_sent)
                sent_tokens = self.bert_tokenizer.tokenize(disease_sent)
                sent_tokens = [self.bert_tokenizer.cls_token] + sent_tokens + [self.bert_tokenizer.sep_token]
                input_ids, attention_masks = self.convert_to_ids(sent_tokens=sent_tokens)
                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_masks)
        return input_ids_list, attention_masks_list, train_to_sent

    def convert_to_ids(self, sent_tokens, label_list=None):
        # convert to id
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(sent_tokens)
        attention_masks = [1] * len(input_ids)

        # pad
        input_ids.extend([0] * (self.max_len - len(input_ids)))
        attention_masks.extend([0] * (self.max_len - len(attention_masks)))
        if len(input_ids) > self.max_len:
            print(sent_tokens)
        assert len(input_ids) == self.max_len
        assert len(attention_masks) == self.max_len
        if label_list is not None:
            label_ids = [self.token_labels.index(l) for l in label_list]
            label_ids.extend([self.token_labels.index(self.pad_label)] * (self.max_len - len(label_ids)))
            assert len(label_ids) == self.max_len
            return input_ids, attention_masks, label_ids
        return input_ids, attention_masks

    def get_disease(self, sent: str) -> List[str]:
        seg_list = self.cut_tokenizer.cut(sent)
        res_list = []
        index = sent.find("@")
        if index != -1:
            title = sent[0:index]
            if "。" not in title:
                res_list.append(title)
        for seg in seg_list:
            if seg in self.disease_list:
                res_list.append(seg)
        return list(set(res_list))


class RelationPredictDataProcess(prepare_data.DataProcess):
    subject_begin = "<s_b>"
    subject_end = "<s_e>"
    object_begin = "<o_b>"
    object_end = "<o_e>"
    special_tokens = {'additional_special_tokens': [subject_begin, subject_end, object_begin, object_end]}
    be_label = "B"
    fl_label = "I"
    nor_label = "O"
    pad_label = 'P'
    token_labels = [pad_label, nor_label, be_label, fl_label]

    def __init__(self, pretrained_path, raw_data_path, output_dir, max_len=512, is_test=True, test_data_path=None):
        super().__init__(pretrained_path, raw_data_path, output_dir, max_len=max_len, is_test=is_test,
                         test_data_path=test_data_path)
        self.bert_tokenizer.add_special_tokens(self.special_tokens)

    @staticmethod
    def schemas_init(raw_data_path):
        schemas = []
        with codecs.open(raw_data_path + "53_schemas.json", encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line == "":
                    break
                schema = json.loads(line)
                if schema['subject_type'] == "疾病":
                    schemas.append("_".join([schema['subject_type'], schema['predicate'], schema['object_type']]))
        return schemas

    def instance_process(self, instance):
        # TODO
        # if len(token_list) > self.bert_tokenizer.model_max_length: pass  # BERT有输入上限
        input_ids = []
        attention_masks = []
        labels = []
        for spo in instance['spo_list']:
            if spo['subject_type'] == "疾病":
                sent = instance['text'].replace(spo['object']['@value'],
                                                self.object_begin + spo['object']['@value'] + self.object_end).replace(
                    spo['subject'], self.subject_begin + spo['subject'] + self.subject_end)
                encoded_dict = self.bert_tokenizer(sent, add_special_tokens=True, padding='max_length',
                                                   max_length=self.max_len)
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
                label = '_'.join([spo['subject_type'], spo['predicate'], spo['object_type']['@value']])
                labels.append(self.schemas.index(label))
        return input_ids, attention_masks, labels

    def test_data_process(self, t_label_datas: torch.Tensor, labeled_to_sent):
        train_to_sent = dict()
        train_index = 0
        input_ids_list = []
        attention_masks_list = []
        for i, token_label_data in enumerate(t_label_datas):
            sent_index, disease = labeled_to_sent[i]
            token_labels = token_label_data.softmax(-1)
            sent = self.test_data[sent_index]['text']
            disease_sent = sent.replace(disease, self.subject_begin + disease + self.subject_end)
            # disease_sent = util.flush_text(disease_sent)
            sent_tokens = self.bert_tokenizer.tokenize(disease_sent)
            sent_tokens = [self.bert_tokenizer.cls_token] + sent_tokens + [self.bert_tokenizer.sep_token]
            objs = self.get_objs(sent_tokens, token_labels)
            for obj_word in objs:
                if obj_word not in sent:
                    obj_word = util.get_showed_word(sent, obj_word)
                train_to_sent[train_index] = (sent_index, disease, obj_word)
                train_index += 1
                labeled_sent = sent.replace(obj_word, self.object_begin + obj_word + self.object_end).replace(
                    disease, self.subject_begin + disease + self.subject_end)
                encoded_dict = self.bert_tokenizer(labeled_sent, add_special_tokens=True, padding='max_length',
                                                   max_length=self.max_len)
                input_ids_list.append(encoded_dict['input_ids'])
                attention_masks_list.append(encoded_dict['attention_mask'])
        return input_ids_list, attention_masks_list, train_to_sent  # Dict[str: Set[int, str, str]]

    def get_objs(self, sent_tokens: List[str], token_labels: torch.Tensor):
        objs = []
        for token_index in range(1, len(sent_tokens) - 1):
            label = token_labels[token_index].argmax(dim=-1)
            if label == 2:  # B
                token_values = [sent_tokens[token_index]]
                inner_index = token_index + 1
                while inner_index < len(sent_tokens) - 2:
                    inner_label = token_labels[inner_index].argmax(dim=-1)
                    if inner_label == 3:  # I
                        token_values.append(util.flush_token(sent_tokens[inner_index]))
                    else:
                        for t in token_values:
                            if t in self.special_tokens['additional_special_tokens']:
                                token_values = []
                                print("Error: token_labels")
                                break
                        if len(token_values) >= 2:
                            objs.append("".join(token_values))
                        break
        return list(set(objs))


if __name__ == '__main__':
    jieba.load_userdict("./dictionary/sub_obj.txt")
    sentt = "（三）产前检查 NTDs的产前检查内容主要包括羊水、母亲血清甲胎蛋白（AFP）检测及B超检查等。胎儿有NTDs可使羊水AFP水平明显升高，同时母亲血清AFP水平也升高。"
    seg_lists = jieba.cut(sentt)
    print(", ".join(seg_lists))
