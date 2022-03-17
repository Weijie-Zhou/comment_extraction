import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import ElectraTokenizer, ElectraModel

from bank_ner.ner.focal_loss import FocalLoss
from bank_ner.ner.models.bert_crf.crf_layer import CRF


class BertCRFModel(nn.Module):
    def __init__(self,
                 bert_base_model_dir,
                 label_size,
                 drop_out_rate=0.5,
                 loss_type='crf_loss',
                 focal_loss_gamma=2,
                 focal_loss_alpha=None): # 增加focal loss参数
        super().__init__()
        # 定义序列标注标签类型数量
        self.label_size =label_size
        # 确保loss_type合法
        assert loss_type in ('crf_loss', 'cross_entropy_loss', 'focal_loss')
        if focal_loss_alpha:
            # 确保focal_loss_alpha合法，必须是一个label的概率分布，因此alpha必须是一个数组，并且长度等于label_size
            assert isinstance(focal_loss_alpha, list) and len(focal_loss_alpha) == label_size
        # 添加focal loss的相关参数到实例变量中
        self.loss_type = loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        # 定义设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载bert分词器、模型
        if 'albert' in bert_base_model_dir.lower():
            # 注意albert base使用bert tokenizer，参考https://huggingface.co/voidful/albert_chinese_base
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = AlbertModel.from_pretrained(bert_base_model_dir)
        elif 'electra' in bert_base_model_dir.lower():
            self.bert_tokenizer = ElectraTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = ElectraModel.from_pretrained(bert_base_model_dir)
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model_dir)
            self.bert_model = BertModel.from_pretrained(bert_base_model_dir)
        # 定义dropout比例
        self.dropout = nn.Dropout(drop_out_rate)
        # 定义线性层，将bert模型输出转成label的概率分布
        self.linear = nn.Linear(self.bert_model.config.hidden_size, label_size)
        # 定义crf层
        self.crf = CRF(label_size)

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
        # 针对不同模型获得模型输出返回
        if isinstance(self.bert_model, ElectraModel):
            last_hidden_state, = bert_out
        else:
            last_hidden_state, pooler_output = bert_out['last_hidden_state'], bert_out['pooler_output']
        # 过dropout层
        seq_outs = self.dropout(last_hidden_state)
        # 获得label概率分布
        logits = self.linear(seq_outs)
        # 获得每个文本的长度，便于loss计算
        lengths = attention_mask.sum(axis=1)
        # # # 根据loss_type, 选择使用维特比解码或者直接argmax
        if self.loss_type == 'crf_loss':
            # 如果是crf loss，则使用维特比解码，通过logits以及lengths，获得最优路径
            best_paths = self.crf.get_batch_best_path(logits, lengths)
        else:
            # 如果是cross_loss,或者focal_loss,则根据篱笆网络(即logits)里面的，当前时刻的所有标签的可能性，通过argmax，取得概率最高的标签
            best_paths = torch.argmax(logits, dim=-1)
        # # data parallel必须确保返回值在gpu
        best_paths = best_paths.to(self.device)
        if labels is not None:
            # # # 根据不同的loss_type，选择不同的loss计算
            # 通过attention_mask来忽略pad
            active_loss = attention_mask.view(-1) == 1
            active_logits, active_labels = logits.view(-1, self.label_size)[active_loss], labels.view(-1)[active_loss]
            if self.loss_type == 'crf_loss':
                # 计算loss时，忽略[CLS]、[SEP]以及PAD部分;
                # 1: 为序列长度，即将CLS截断； lengths-2：根据序列原始长度(忽略了PAD)-CLS-SEP
                loss = self.crf.negative_log_loss(inputs=logits[:, 1:, :], length=lengths - 2, tags=labels[:, 1:])
            elif self.loss_type == 'cross_entropy_loss':
                # 使用交叉熵损失, ignore_index=-1: label=1被忽略
                loss = CrossEntropyLoss(ignore_index=-1)(active_logits, active_labels)
            else:
                # 使用focal_loss
                # 进一步忽略-1的部分，即[CLS]、[SEP]
                active_loss = active_labels != -1
                active_logits, active_labels = active_logits[active_loss], active_labels[active_loss]
                loss = FocalLoss(gamma=self.focal_loss_gamma,
                                 alpha=self.focal_loss_alpha)(active_logits, active_labels)
            return best_paths, loss
        # 直接返回预测的labels
        return best_paths

    def get_bert_tokenizer(self):
        # 返回bert分词器
        return self.bert_tokenizer