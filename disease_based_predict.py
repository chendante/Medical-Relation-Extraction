from transformers import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import util
import preparation.entity_extraction_first as pr
from run_multi_label import BertForMultiLabelSequenceClassification

test_data_path = "./raw_data/test1.json"

if __name__ == '__main__':
    # prepare data
    data_processor = pr.DiseaseBasedDataProcess(pretrained_path="./pretrained_model/bert_wwm/",
                                                raw_data_path="./raw_data/", output_dir="./",
                                                split_dic="./preparation/dictionary/sub_obj.txt",
                                                voc_type_path="./preparation/dictionary/sub_obj_type.json",
                                                test_data_path=test_data_path)
    input_ids, attention_masks, labeled_to_sent = data_processor.test_data_process()
    input_ids = torch.from_numpy(input_ids).type(torch.long)
    attention_masks = torch.from_numpy(attention_masks).type(torch.long)

    token_prediction_data = TensorDataset(input_ids, attention_masks)
    token_prediction_dataloader = DataLoader(token_prediction_data, batch_size=32)

    token_model = BertForTokenClassification.from_pretrained(
        "./model_save/disease_based_token/",
        # num_labels=4,
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )
    print(token_model.num_labels)
    token_model.cuda()
    token_model.eval()

    predictions = []
    for batch in token_prediction_dataloader:
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()

        with torch.no_grad():
            logits = token_model(b_input_ids, attention_mask=b_input_mask)
        logits = logits.detach().cpu()
        predictions.append(logits)
    predictions = torch.cat(predictions)
    print("     DONE.")

    data_processor = pr.RelationPredictDataProcess(pretrained_path="./pretrained_model/bert_wwm/",
                                                   raw_data_path="./raw_data/", output_dir="./",
                                                   test_data_path=test_data_path)
    input_ids, attention_masks, train_to_sent = data_processor.test_data_process(predictions, labeled_to_sent)

    input_ids = torch.from_numpy(np.array(input_ids, dtype=np.long)).type(torch.long)
    attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.long)).type(torch.long)

    cls_prediction_data = TensorDataset(input_ids, attention_masks)
    cls_prediction_dataloader = DataLoader(cls_prediction_data, batch_size=32)

    cls_model = BertForSequenceClassification.from_pretrained(
        "./model_save/disease_based_cls/",
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )
    print(cls_model.num_labels)
    cls_model.cuda()
    cls_model.eval()

    cls_predictions = []
    for batch in cls_prediction_dataloader:
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        with torch.no_grad():
            outputs = cls_model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu()
        cls_predictions.append(logits)
    cls_predictions = torch.cat(cls_predictions)

    print("     DONE.")

    for i, cls_prediction in enumerate(cls_predictions):
        sent_index, disease, obj_word = train_to_sent[i]

