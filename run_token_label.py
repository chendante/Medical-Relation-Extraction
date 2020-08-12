from transformers import *
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import util

if __name__ == '__main__':
    model = BertForTokenClassification.from_pretrained(
        "./pretrained_model/bert_wwm/",
        num_labels=6,
        output_attentions=False,  # 模型是否返回 attentions weights.
        output_hidden_states=False,  # 模型是否返回所有隐层状态.
    )
    # model.cuda()
    output_dir = "./model_save/token_label/"
    data_path = "./preparation/processed_data/token_label/"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    input_ids = torch.from_numpy(np.load(data_path + "input_ids.npy")).type(torch.long)
    attention_masks = torch.from_numpy(np.load(data_path + "attention_masks.npy")).type(torch.long)
    labels = torch.from_numpy(np.load(data_path + "labels.npy")).type(torch.float)
    token_type_ids = torch.from_numpy(np.load(data_path + "token_type_ids.npy"))
    dataset = TensorDataset(input_ids, attention_masks, labels, token_type_ids)

    # 训练 epochs。 BERT 作者建议在 2 和 4 之间，设大了容易过拟合
    epochs = 4
    batch_size = 2

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )

    # 总的训练样本数
    total_steps = len(train_dataloader) * epochs

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    total_t0 = time.time()

    # 将模型设置为训练模式。这里并不是调用训练接口的意思
    model.train()
    device = torch.device('cpu')

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
            b_token_type_ids = batch[3].to(device)

            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()

            # 前向传播
            loss = model(b_input_ids,
                         token_type_ids=b_token_type_ids,
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

    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)