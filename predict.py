from transformers import *
import torch


def main():
    model = BertForSequenceClassification.from_pretrained("./model_save/")



if __name__ == '__main__':
    main()
