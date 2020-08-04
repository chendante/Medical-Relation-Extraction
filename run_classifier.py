from transformers import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch


def main():
    model = BertForSequenceClassification.from_pretrained("./pretrained_model/bert_wwn_ext/", num_labels=53)
    input_ids = torch.cat(np.load("./processed_data/input_ids.npy"), 0)
    attention_masks = torch.cat(np.load("./processed_data/attention_masks.npy"))
    labels = torch.cat(np.load("./processed_data/labels.npy"))
    dataset = TensorDataset(input_ids, attention_masks, labels)
    model.cuda()

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



if __name__ == "__main__":
    main()
