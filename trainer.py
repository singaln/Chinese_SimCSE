# coding=utf-8
# @Time:2021/6/2310:21
# @author: SinGaln

import os
import torch
import logging
import numpy as np
from tqdm import tqdm, trange
from torch.optim import lr_scheduler
from .utils import simcse_loss, model_class, get_device
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_data=None, dev_data=None, test_data=None):
        self.args = args
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.config_class, self.model_class, _ = model_class[args.model_type]
        self.configs = self.config_class.from_pretrained(args.bert_model_path,
                                                         finetuning_task=self.args.task)
        self.model = self.model_class.from_pretrained(args.bert_model_path,
                                                      config=self.configs,
                                                      args=args)

        self.device = get_device(args)
        self.model.to(self.device)
        # 多GPU
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def train(self):
        train_sampler = RandomSampler(self.train_data)
        train_loader = DataLoader(self.train_data, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            total_steps = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                len(train_loader) // self.args.gradient_accumulation_steps) + 1
        else:
            total_steps = len(train_loader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # optimizer and schedule
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params":[p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay":self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        lr_decay = lr_scheduler.StepLR(optimizer, step_size= 100, gamma=1e-3)
        schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                   num_training_steps=total_steps)
        # train information
        logger.info("********** Running Training **********")
        logger.info("num example = %d", len(self.train_data))
        logger.info("num epochs = %d", self.args.num_train_epochs)
        logger.info("train batch size = %d", self.args.train_batch_size)
        logger.info("gradient accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("total steps = %d", total_steps)
        logger.info("logger steps = %d", self.args.logging_steps)
        logger.info("save steps = %d", self.args.save_steps)

        global_steps = 0
        total_loss = 0.0
        # loss_func = torch.nn.CrossEntropyLoss()
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_loader)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs_ids, attention_mask, token_type_ids, label_id = batch

                outputs = self.model(input_ids=inputs_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                # print("outputs", outputs)
                # print("label_id", label_id, label_id.shape)
                loss = simcse_loss(outputs)

                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                else:
                    loss.backward()
                total_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    schedule.step()
                    lr_decay.step()
                    optimizer.zero_grad()
                    global_steps += 1
                    if self.args.logging_steps > 0 and global_steps % self.args.logging_steps == 0:
                        self.evaluate("dev")
                    if self.args.save_steps > 0 and global_steps % self.args.save_steps == 0:
                        self.save_model()
                if 0 < self.args.max_steps < global_steps:
                    epoch_iterator.close()
                    break
        return global_steps, total_loss / global_steps

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_data
        elif mode == "dev":
            dataset = self.dev_data
        else:
            raise Exception("The dataset is not existing!")

        eval_sampler = SequentialSampler(dataset)
        eval_loader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # eval logging
        logger.info("********** logger information **********")
        logger.info("num example = %d", len(dataset))
        logger.info("batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        eval_steps = 0
        label_preds = []
        label_ids = None

        correct = torch.nn.Softmax(dim=1)
        self.model.eval()
        for batch in tqdm(eval_loader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, token_type_ids, label_id = batch
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                regular_label = correct(outputs)
                preds = regular_label.cpu().detach().numpy()
                pred_label = np.argmax(preds, axis=1)
                corr_lst = torch.equal(label_id, torch.tensor(pred_label))
                accuracy = corr_lst / len(label_id)
                return accuracy


    def save_model(self):
        # 模型保存
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # 保存模型训练的超参
        torch.save(self.args, os.path.join(self.args.model_dir, "train_args.bin"))
        logger.info("model parameters save %s", self.args.model_dir)

    def load_model(self):
        # 加载模型
        if os.path.exists(self.args.model_dir):
            raise Exception("The model is not existing!")
        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args)
            self.model.to(self.device)
            logger.info("********** model load success **********")
        except:
            raise Exception("The model lost or damage!")