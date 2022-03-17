import json
import math

import numpy as np
import torch
from sklearn.utils import shuffle
from transformers import AdamW
from torch.nn import DataParallel

from bank_ner.ner import logger
from bank_ner.ner.models.base.base_trainer import BaseTrainer
from bank_ner.ner.models.bert_crf.bert_crf_model import BertCRFModel
from bank_ner.ner.models.vocab import Vocab
from sklearn.metrics import f1_score, accuracy_score


class BertCRFTrainer(BaseTrainer):
    def __init__(self,
                 pretrained_model_dir,
                 model_dir,
                 learning_rate=1e-3,
                 ckpt_name='bert_model.bin',
                 vocab_name='vocab.json',
                 enable_parallel=False, # 增加是否启用并行参数
                 loss_type='crf_loss',
                 focal_loss_gamma=2,
                 focal_loss_alpha=None # 增加focal_loss相关参数
                 ):
        # 预训练语言模型路径
        self.pretrained_model_dir = pretrained_model_dir
        # 模型保存路径
        self.model_dir = model_dir
        # 模型保存名称
        self.ckpt_name = ckpt_name
        # 词典保存名称
        self.vocab_name = vocab_name
        # # 设置是否启用并行参数为实例变量
        self.enable_parallel = enable_parallel
        # # # 设置focal_loss相关参数
        self.loss_type = loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        # 定义设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 定义学习率
        self.learning_rate = learning_rate
        # 预定义batch_size, epoch，在train函数中进行传参
        self.batch_size = None
        self.epoch = None
        # 初始化词典
        self.vocab = Vocab()

    def _build_model(self):
        '''构建bert-crf模型'''
        # 初始化bert-crf模型
        # 传入focal_loss相关参数，来初始化模型
        self.model = BertCRFModel(self.pretrained_model_dir,
                                  self.vocab.label_size,
                                  loss_type=self.loss_type,
                                  focal_loss_gamma=self.focal_loss_gamma,
                                  focal_loss_alpha=self.focal_loss_alpha)
        # 设置AdamW优化器
        # 设置bias和LayerNormb不使用正则化
        no_decay = ['bias', 'LayerNorm.weight']
        # 区分bert层的参数和crf层的参数
        bert_parameters = [(name, param) for name, param in self.model.named_parameters() if 'crf' not in name]
        crf_parameters = [(name, param) for name, param in self.model.named_parameters() if 'crf' in name]
        # 定义参数正则化: 权重衰减, crf层使用较大的学习率
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in crf_parameters], 'lr': self.learning_rate * 10},  # crf层使用较大的学习率
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        # 使用bert的vocab更新我们的Vocab词典对象，label需要自己构建，词使用bert提供的词典
        self.vocab.set_vocab2id(self.model.get_bert_tokenizer().vocab)
        self.vocab.set_id2vocab({_id: char for char, _id in self.vocab.vocab2id.items()})
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])
        # # 启动并行，使用DataParallel封装model
        if self.enable_parallel:
            # device_ids：设备id，即有多少gpu设备
            # 启动并行，model就被封装为parallel对象，需要额外通过调用module对象来获得tokenizer或者state_dict
            self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        # 将模型拷贝到设备上
        self.model.to(self.device)

    def _save_config(self):
        '''保存训练参数'''
        config = {
            'vocab_size': self.vocab.vocab_size,
            'label_size': self.vocab.label_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'ckpt_name': self.ckpt_name,
            'vocab_name': self.vocab_name,
            # # 将并行参数以及focal_loss参数保存
            'enable_parallel': self.enable_parallel,
            'loss_type': self.loss_type,
            'focal_loss_gamma': self.focal_loss_gamma,
            'focal_loss_alpha': self.focal_loss_alpha
        }
        with open('{}/train_config.json'.format(self.model_dir), 'w') as f:
            f.write(json.dumps(config, indent=4))

    def _transform_batch(self, batch_texts, batch_labels, max_length=512):
        '''将batch的文本及labels转换为bert的输入tensor形式'''
        batch_input_ids, batch_att_mask, batch_label_ids = [], [], []
        for text, labels in zip(batch_texts, batch_labels):
            # 判断遍历的text文本的格式应当为list
            assert isinstance(text, list)
            # 确保输入encode_plus函数的内容为str格式，自己分词，不使用bert进行分词，因为bert分词可能会出现问题
            text = ' '.join(text)
            # # 根据是否并行，获取bert_tokenizer
            bert_tokenizer = self.model.bert_tokenizer if not self.enable_parallel else self.model.module.bert_tokenizer
            encoded_dict = bert_tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
                truncation=True
            )
            batch_input_ids.append(encoded_dict['input_ids'])
            batch_att_mask.append(encoded_dict['attention_mask'])
            batch_label_ids.append(
                # [CLS] 与 [SEP] 用-1填充，尾部用pad填充(减去2，是由于cls、sep的占位)
                [-1] + [self.vocab.tag2id[_label] for _label in labels] + [-1] + [self.vocab.pad_tag_id] * (max_length - len(labels) - 2)
            )
        # 将batch_input_ids, batch_att_mask, batch_label_ids转为tensor
        batch_input_ids = torch.cat(batch_input_ids)
        batch_att_mask = torch.cat(batch_att_mask)
        batch_label_ids = torch.LongTensor(batch_label_ids)
        # 将batch_input_ids, batch_att_mask, batch_label_ids拷贝到设备上
        batch_input_ids, batch_att_mask, batch_label_ids = \
            batch_input_ids.to(self.device), batch_att_mask.to(self.device), batch_label_ids.to(self.device)
        return batch_input_ids, batch_att_mask, batch_label_ids

    def _get_acc_f1_one_step(self, labels_predict_batch, labels_batch):
        '''直接根据预测的labels和实际的labels计算acc、f1（而不是根据logits）'''
        # total, correct = 0, 0
        # for labels_predict, labels in zip(labels_predict_batch, labels_batch):
        #     # 去除pad部分，以及[CLS]、[SEP] token
        #     active_labels_predict = labels_predict[labels != self.vocab.pad_tag_id][1:-1]
        #     active_labels = labels[labels != self.vocab.pad_tag_id][1: -1]
        #     total += len(active_labels)
        #     correct += (active_labels_predict.cpu() == active_labels.cpu()).sum().item()
        # accuracy = correct / total
        # return float(accuracy)

        true_labels = []
        predict_labels = []
        for labels_predict, labels in zip(labels_predict_batch, labels_batch):
            # 去除pad部分，以及[CLS]、[SEP] token
            active_labels_predict = labels_predict[labels != self.vocab.pad_tag_id][1:-1]
            active_labels = labels[labels != self.vocab.pad_tag_id][1: -1]
            true_labels += active_labels.cpu().numpy().tolist()
            predict_labels += active_labels_predict.cpu().numpy().tolist()
        accuracy = accuracy_score(true_labels, predict_labels)
        f1 = f1_score(true_labels, predict_labels, average='weighted')
        return accuracy, f1



    def train(self, train_texts, labels, validate_texts, validate_labels, batch_size=30, epochs=10):
        '''
        模型训练
        :param train_texts: list[list[str]]. 训练集样本
        :param labels: list[list[str]]. 训练集标签
        :param validate_texts: list[list[str]]. 验证集样本
        :param validate_labels: list[list[str]]. 验证集标签
        :param batch_size: int
        :param epochs: int
        :return:
        '''
        # 设置batch_size、epoch参数
        self.batch_size = batch_size
        self.epoch = epochs
        # 构建词典，只构建labels，词库使用bert的
        self.vocab.build_vocab(labels=labels, build_texts=False)
        # 构建bert-crf模型
        self._build_model()
        # 保存词典
        self.vocab.save_vocab('{}/{}'.format(self.model_dir, self.vocab_name))
        # 保存训练参数
        self._save_config()
        logger.info('train samples: {}, validate samples: {}'.format(len(train_texts), len(validate_texts)))
        print('setp_nums/train_epoch: ', math.ceil(len(train_texts) / batch_size))
        # 初始化最优f1
        best_f1 = -float('inf')
        # 初始化训练步数
        step = 0
        for epoch in range(epochs):
            for batch_idx in range(math.ceil(len(train_texts) / batch_size)):
                # 迭代batch
                text_batch = train_texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]
                labels_batch = labels[batch_size * batch_idx: batch_size * (batch_idx + 1)]
                step += 1
                # 设置为训练模式
                self.model.train()
                # 梯度归零
                self.model.zero_grad()
                # 获得当前batch的最大文本长度，长度+2的原因是要加上[CLS]和[SEP]
                batch_max_len = max([len(text) for text in text_batch]) + 2
                # 将batch转成bert需要的输入格式
                batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(
                    text_batch, labels_batch, max_length=batch_max_len
                )
                # 获得bert-crf模型输出
                best_paths, loss = self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
                # # 如果启动并行，需将多张卡返回的sub-batch loss相加，得到总batch的loss
                if self.enable_parallel:
                    # 在启用了并行之后，best_paths其实是多个设备的返回，拼接成的列表，这个不需要我们进行操作
                    # 但是loss则是多个设备的返回，拼接成的loss列表，但是我们对loss反向传播仅需要一个总和的loss，因此需要sum
                    loss = loss.sum()
                # 反向传播, 更新参数
                loss.backward()
                self.optimizer.step()
                # 获得当前step的训练集的准确率
                train_acc, trian_f1 = self._get_acc_f1_one_step(best_paths, batch_label_ids)
                logger.info(
                    'epoch %d, step %d, train loss %.4f, train acc %.4f, train f1 %.4f' % (
                        epoch, step, loss, train_acc, trian_f1)
                )

            # 获得验证集的准确率、f1
            valid_acc, valid_f1, valid_loss = self.validate(validate_texts, validate_labels)
            logger.info(
                'epoch %d, step %d, valid loss %.4f valid acc %.4f, valid f1 %.4f'% (
                    epoch, step, valid_loss, valid_acc, valid_f1)
            )
            if valid_f1 > best_f1:
                # 保存模型
                best_f1 = valid_f1
                # # 根据是否启用并行，获得state_dict
                state_dict = self.model.state_dict() if not self.enable_parallel else self.model.module.state_dict()
                torch.save(state_dict, '{}/{}'.format(self.model_dir, self.ckpt_name))
                logger.info('model saved!')
        logger.info('train finished!')

    def validate(self, batch_texts, batch_labels):
        '''
        使用当前的model评估验证集
        :param batch_texts: list[list[str]] or np.array.验证集样本（原始的或者转换过）
        :param batch_labels: list[list[str]] or np.array.验证集标签（原始的或者转换过）
        :return: float.验证集上acc, loss
        '''
        # 设置为评估模式
        self.model.eval()
        acc_lst = []
        f1_lst = []
        loss_lst = []
        with torch.no_grad(): # 不累积梯度
            # 获得模型需要的输入
            for batch_idx in range(math.ceil(len(batch_texts) / self.batch_size)):
                # 迭代batch
                text_batch = batch_texts[self.batch_size * batch_idx: self.batch_size * (batch_idx + 1)]
                labels_batch = batch_labels[self.batch_size * batch_idx: self.batch_size * (batch_idx + 1)]
                batch_max_len = max([len(text) for text in text_batch]) + 2
                batch_input_ids, batch_att_mask, batch_label_ids = self._transform_batch(
                    text_batch, labels_batch, max_length=batch_max_len
                )
                # 获得模型输出
                best_paths, loss = self.model(batch_input_ids, batch_att_mask, labels=batch_label_ids)
                # 如果启用并行，需将多张卡返回的sub-batch loss相加，得到总batch的loss
                if self.enable_parallel:
                    loss = loss.sum()
                # 计算验证集acc
                acc, f1 = self._get_acc_f1_one_step(best_paths, batch_label_ids)
                acc_lst.append(acc)
                f1_lst.append(f1)
                loss_lst.append(loss.item())
            return np.mean(acc_lst), np.mean(f1_lst), np.mean(loss_lst)