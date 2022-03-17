import json
import math

import torch
from torch.nn import DataParallel

from bank_ner.ner.models.base.base_predictor import BasePredictor
from bank_ner.ner.models.bert_crf.bert_crf_model import BertCRFModel
from bank_ner.ner.models.vocab import Vocab
from bank_ner.ner import logger


class BERTCRFPredictor(BasePredictor):
    def __init__(self,
                 pretrained_model_dir,
                 model_dir,
                 vocab_name='vocab.json',
                 # # 增加是否启用并行参数
                 enable_parallel=False
                 ):
        # 获得预训练语言模型目录
        self.pretrained_model_dir = pretrained_model_dir
        # 获得模型保存目录
        self.model_dir = model_dir
        # # 将启用并行参数保存到实例变量中
        self.enable_parallel = enable_parallel
        # 获得设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化词典
        self.vocab = Vocab()
        # 导入config参数
        self._load_config()
        # 加载词典
        self.vocab.load_vocab('{}/{}'.format(model_dir, vocab_name))
        # 加载模型
        self._load_model()
        logger.info('load model success!')

    def _load_config(self):
        # 导入配置参数
        with open('{}/train_config.json'.format(self.model_dir), 'r') as f:
            self._config = json.loads(f.read())

    def _load_model(self):
        # 初始化模型
        self.model = BertCRFModel(self.pretrained_model_dir,
                                  self._config['label_size'],
                                  # # # 传入focal_loss需要的参数
                                  loss_type=self._config['loss_type'],
                                  focal_loss_gamma=self._config['focal_loss_gamma'],
                                  focal_loss_alpha=self._config['focal_loss_alpha']
                                  )
        # 导入训练好的模型参数
        self.model.load_state_dict(
            torch.load('{}/{}'.format(self.model_dir, self._config['ckpt_name']), map_location=self.device)
        )
        # 设置为评估模式
        self.model.eval()
        # # 根据启用并行参数，使用DataParallel封装model
        if self.enable_parallel:
            self.model = DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        # 将模型拷贝到设备上
        self.model.to(self.device)
        # 词典设置unk、pad token
        self.vocab.set_unk_vocab_id(self.vocab.vocab2id['[UNK]'])
        self.vocab.set_pad_vocab_id(self.vocab.vocab2id['[PAD]'])

    def predict(self, texts, batch_size=64, max_len=512):
        '''
        模型预测
        :param texts: list[list[str]].预测样本
        :param batch_size: int
        :param max_len: int.最大序列长度（请和bert预训练模型中的max_position_embeddings保持一致）
        :return: list[list[str]].标签序列
        '''
        # 存储预测出的标签
        batch_labels = []
        for batch_idx in range(math.ceil(len(texts) / batch_size)):
            # 迭代样本
            text_batch = texts[batch_size * batch_idx: batch_size * (batch_idx + 1)]
            # 获得当前batch中文本的最大长度
            batch_max_len = min(max([len(text) for text in text_batch]) + 2, max_len)
            # 将batch数据转成模型需要的输入格式
            batch_input_ids, batch_att_mask = [], []
            for text in text_batch:
                # 迭代每条文本
                # 判断文本为list格式
                assert isinstance(text, list)
                # 将text转成str格式，自己分词，不使用bert进行分词，因为bert分词可能会出现问题
                text = ' '.join(text)
                # # 根据是否并行，获取bert_tokenizer
                bert_tokenizer = self.model.bert_tokenizer if not self.enable_parallel else self.model.module.bert_tokenizer
                # 使用分词器，将文本转成token
                encoded_dict = bert_tokenizer.encode_plus(
                    text, max_length=batch_max_len, padding='max_length', return_tensors='pt', truncation=True)
                batch_input_ids.append(encoded_dict['input_ids'])
                batch_att_mask.append(encoded_dict['attention_mask'])
            # 将token转成tensor格式
            batch_input_ids = torch.cat(batch_input_ids)
            batch_att_mask = torch.cat(batch_att_mask)
            # 将batch_input_ids、batch_att_mask拷贝到设备上
            batch_input_ids, batch_att_mask = batch_input_ids.to(self.device), batch_att_mask.to(self.device)
            with torch.no_grad():
                best_paths = self.model(batch_input_ids, batch_att_mask)
                for best_path, att_mask in zip(best_paths, batch_att_mask):
                    # 去除掉pad部分，以及[CLS]、[SEP] token
                    active_labels = best_path[att_mask == 1][1:-1]
                    labels = [self.vocab.id2tag[label_id.item()] for label_id in active_labels]
                    batch_labels.append(labels)

        return batch_labels

    def get_bert_tokenizer(self):
        # 返回bert的分词器
        return self.model.bert_tokenizer