from transformers import *


def main():
    model = BertForSequenceClassification.from_pretrained("./pretrained_model/bert_wwn_ext/", num_labels=53)


if __name__ == "__main__":
    main()