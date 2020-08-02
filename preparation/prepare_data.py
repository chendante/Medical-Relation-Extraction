import json
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("../pretrained_model/")


class DataProcess(object):
    def __init__(self, pretrained_path, raw_data_path, is_test=True):
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)



def main():
    t = bert_tokenizer.tokenize("痔@肛镜检查方法简便安全，可全方位观察肛管及所有痔组织。痔@另一种替代的方法为纤维内窥镜反转观察，此法操作要求高，需更高的技巧水平。")
    t = bert_tokenizer.cls_token
    print(t)


if __name__ == "__main__":
    main()
