import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel

from focal_loss import FocalLoss


class BertClassificationModel(nn.Module):
    '''基于bert的文本分类模型'''
    def __init__(self,
                 bert_base_model_dir,
                 label_size,
                 device,
                 drop_out_rate=0.5,
                 loss_type='focal_loss',
                 # focal loss参数
                 focal_loss_gamma=2,
                 focal_loss_alpha=None):
        super().__init__()
        # 定义序列标注标签类型数量
        self.label_size =label_size
        # 确保传入的loss_type合法
        assert loss_type in ('cross_entropy_loss', 'focal_loss')
        if focal_loss_alpha:
            # 确保focal_loss_alpha合法，必须是一个label的概率分布，因此alpha必须是一个数组，并且长度等于label_size
            assert isinstance(focal_loss_alpha, list) and len(focal_loss_alpha) == label_size
        # 添加focal loss的相关参数到实例变量中
        self.loss_type = loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        # 定义设备
        self.device = device
        # 加载bert模型
        self.bert_model = BertModel.from_pretrained(bert_base_model_dir)
        # 定义dropout比例
        self.dropout = nn.Dropout(drop_out_rate)
        # 定义线性层，将bert模型输出转成label的概率分布
        self.linear = nn.Linear(self.bert_model.config.hidden_size, label_size)

    def forward(self,
                input_ids, # 文本转成的token_id
                attention_mask=None, # attention的mask
                token_type_ids=None, # 默认为全0
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None
                ):
        # 获得bert模型输出
        bert_out = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=None
        )
        pooler_output = bert_out['pooler_output']
        # 过dropout层
        seq_outs = self.dropout(pooler_output)
        # 获得label概率分布
        logits = self.linear(seq_outs)
        loss = None
        if labels is not None:
            # 根据不同的loss_type，选择不同的loss计算
            if self.loss_type == 'cross_entropy_loss':
                # 使用交叉熵损失, ignore_index=-1: label=1被忽略
                loss = CrossEntropyLoss(ignore_index=-1)(logits, labels)
            else:
                loss = FocalLoss(gamma=self.focal_loss_gamma,
                                 alpha=self.focal_loss_alpha)(logits, labels)
        return loss, logits

    def get_bert_tokenizer(self):
        # 返回bert分词器
        return self.bert_tokenizer