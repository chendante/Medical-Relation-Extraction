from transformers import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import util
from preparation import prepare_data
from run_multi_label import BertForMultiLabelSequenceClassification
import codecs
import json


def flush(token: str):
    if token.startswith("##"):
        return token[2:]
    return token


def main():
    # Get relation type values
    data_processor = prepare_data.MultiLabelDataProcess(pretrained_path="./pretrained_model/bert_wwm/",
                                                        raw_data_path="./raw_data/", output_dir="./",
                                                        test_data_path="./raw_data/val_data.json")
    input_ids, attention_masks = data_processor.test_data_process()
    input_ids = torch.from_numpy(input_ids).type(torch.long)
    attention_masks = torch.from_numpy(attention_masks).type(torch.long)

    ml_prediction_data = TensorDataset(input_ids, attention_masks)
    ml_prediction_dataloader = DataLoader(ml_prediction_data, batch_size=32)
    ml_model = BertForMultiLabelSequenceClassification.from_pretrained(
        "./model_save/multi_label_s_32/",
        # num_labels=53,
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )
    ml_model.cuda()
    ml_model.eval()
    predictions = []
    for batch in ml_prediction_dataloader:
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()

        with torch.no_grad():
            logits = ml_model(b_input_ids, attention_mask=b_input_mask)
        logits = logits.detach().cpu()
        predictions.append(logits)
    predictions = torch.cat(predictions)
    print("Relation Type Predict     DONE.")

    # Get Token Label
    data_processor = prepare_data.LabelDataProcess(pretrained_path="./pretrained_model/bert_wwm/",
                                                   raw_data_path="./raw_data/", output_dir="./",
                                                   test_data_path="./raw_data/val_data.json")
    input_ids, attention_masks, token_type_ids, train_to_sent = data_processor.test_data_process(predictions, 0.5)
    input_ids = torch.from_numpy(np.array(input_ids, dtype=np.long)).type(torch.long)
    attention_masks = torch.from_numpy(np.array(attention_masks, dtype=np.long)).type(torch.long)
    token_type_ids = torch.from_numpy(np.array(token_type_ids, dtype=np.long)).type(torch.long)

    tl_prediction_data = TensorDataset(input_ids, attention_masks, token_type_ids)
    tl_prediction_dataloader = DataLoader(tl_prediction_data, batch_size=32)
    tl_model = BertForTokenClassification.from_pretrained(
        "./model_save/token_label_s_32/",
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )
    tl_model.cuda()
    tl_model.eval()
    tl_predictions = []
    for batch in tl_prediction_dataloader:
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_token_type_ids = batch[2].cuda()
        with torch.no_grad():
            outputs = tl_model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)
        logits = outputs[0]
        logits = logits.detach().cpu()
        tl_predictions.append(logits)
    tl_predictions = torch.cat(tl_predictions)

    print("Token Label Predict     DONE.")
    entity_relation_pairs = []
    for i, pd in enumerate(tl_predictions):
        s_index, r_label = train_to_sent[i]
        entity_relation_pairs.append({"relation_label": r_label, "s_index": s_index, "obj": [], "sub": []})
        sent = data_processor.test_data[s_index]['text']
        sent = data_processor.flush_text(sent)
        sent_tokens = data_processor.bert_tokenizer.tokenize(sent)
        sent_tokens = [data_processor.bert_tokenizer.cls_token] + sent_tokens + [data_processor.bert_tokenizer.sep_token]
        p_labels = pd.softmax(-1)
        token_index = 1
        while token_index < len(sent_tokens) - 1:
            label = p_labels[token_index].argmax(dim=-1)
            if label == 2:  # B-SUB
                token_values = [sent_tokens[token_index]]
                while token_index < len(sent_tokens) - 2:
                    token_index += 1
                    label = p_labels[token_index].argmax(dim=-1)
                    if label == 4:  # I-SUB
                        token_values.append(flush(sent_tokens[token_index]))
                    else:
                        entity_relation_pairs[i]["sub"].append("".join(token_values))
                        break
                continue
            if label == 3:  # B-OBJ
                token_values = [flush(sent_tokens[token_index])]
                while token_index < len(sent_tokens) - 2:
                    token_index += 1
                    label = p_labels[token_index].argmax(dim=-1)
                    if label == 5:  # I-OBJ
                        token_values.append(flush(sent_tokens[token_index]))
                    else:
                        entity_relation_pairs[i]["obj"].append("".join(token_values))
                        break
                continue
            token_index += 1

    final_result = [{"text": t_d["text"], "spo_list": []} for t_d in data_processor.test_data]

    for er_pair in entity_relation_pairs:
        fri = er_pair["s_index"]
        sent = final_result[fri]["text"]
        s_num = sent.count("。") + sent.count("？")
        combined = False
        if sent[-1] != '。' and sent[-1] != '？':
            s_num += 1
        if s_num > 1:
            combined = True
        s_type, predicate, o_type = data_processor.schemas[er_pair["relation_label"]].split("_")
        objs = list(set(er_pair["obj"]))
        subs = list(set(er_pair["sub"]))
        for sub in subs:
            for obj in objs:
                final_result[fri]["spo_list"].append({
                    "Combined": combined, "predicate": predicate, "subject": sub, "subject_type": s_type,
                    "object": {"@value": obj},
                    "object_type": {"@value": o_type}
                })

    with codecs.open("./result/result.json", mode='w+', encoding='utf-8') as f:
        for l in final_result:
            f.write(json.dumps(l) + "\n")


if __name__ == '__main__':
    main()
