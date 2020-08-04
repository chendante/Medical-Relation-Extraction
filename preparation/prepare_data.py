import json
from transformers import BertTokenizer
import codecs

bert_tokenizer = BertTokenizer.from_pretrained("../pretrained_model/bert_wwn/")


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
        for t_d in self.train_data:
            


def main():
    t = bert_tokenizer.tokenize("痔@肛镜检查方法简便安全，可全方位观察肛管及所有痔组织。痔@另一种替代的方法为纤维内窥镜反转观察，此法操作要求高，需更高的技巧水平。")
    t = bert_tokenizer.cls_token
    print(t)


if __name__ == "__main__":
    main()
