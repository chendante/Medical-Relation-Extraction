from transformers import *
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import time
import datetime
import os
import util
from preparation import prepare_data

trained_model_path = "./model_save/multi_label/"
pretrained_model_path = "./pretrained_model/bert_wwm/"


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def main():
    model = BertForMultiLabelSequenceClassification.from_pretrained(
        "./pretrained_model/bert_wwm/",
        num_labels=53,
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )

    if not os.path.exists(trained_model_path):
        os.makedirs(trained_model_path)
    input_ids = torch.from_numpy(np.load("./preparation/processed_data/input_ids.npy")).type(torch.long)
    attention_masks = torch.from_numpy(np.load("./preparation/processed_data/attention_masks.npy")).type(torch.long)
    labels = torch.from_numpy(np.load("./preparation/processed_data/labels.npy")).type(torch.float)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    # model.cuda()

    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )

    # 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
    epochs = 4

    # 总的训练样本数
    total_steps = len(train_dataloader) * epochs

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    total_t0 = time.time()
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # 统计单次 epoch 的训练时间
        t0 = time.time()

        # 重置每次 epoch 的训练总 loss
        total_train_loss = 0

        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        model.train()
        device = torch.device('cuda')
        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):

            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = util.format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            loss = model(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)

            # 累加 loss
            total_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

        # 平均训练误差
        avg_train_loss = total_train_loss / len(train_dataloader)

        # 单次 epoch 的训练时长
        training_time = util.format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(util.format_time(time.time() - total_t0)))

    print("Saving model to %s" % trained_model_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(trained_model_path)


def predict():
    # Get test data
    prepare_data.MultiLabelDataProcess(pretrained_path=pretrained_model_path, raw_data_path="./raw_data", output_dir="./", )

    model = BertForMultiLabelSequenceClassification.from_pretrained(
        trained_model_path,
        # num_labels=53,
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )
    print(model.num_labels)
    input_ids = torch.from_numpy(np.load("./preparation/processed_data/input_ids.npy")).type(torch.long)
    attention_masks = torch.from_numpy(np.load("./preparation/processed_data/attention_masks.npy")).type(torch.long)


if __name__ == '__main__':
    main()
