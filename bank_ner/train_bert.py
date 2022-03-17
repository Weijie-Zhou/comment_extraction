import numpy as np
import torch

import evaluate
from bank_ner.ner.models.bert_crf.bert_crf_trainer import BertCRFTrainer
from bank_ner.ner.models.bert_crf.bert_crf_predictor import BERTCRFPredictor


# 设置随机种子
seed = 42
# torch cpu 随机种子
torch.manual_seed(seed)
# torch gpu 随机种子
torch.cuda.manual_seed_all(seed)
# numpy 随机种子
np.random.seed(seed)

def read_conll(file_path):
    # 读取数据
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts, labels = [], []
        for example in f.read().split('\n\n'):
            # 迭代每条样本
            example = example.strip()
            if not example:
                continue
            texts.append([]), labels.append([])
            for term in example.split('\n'):
                if len(term.split('\t')) != 2:
                    # 跳过不合法的行
                    continue
                char, label = term.split('\t')
                texts[-1].append(char), labels[-1].append(label)
        return texts, labels


# 读取数据
train_texts, train_labels = read_conll('./data/ner/train.txt')
dev_texts, dev_labels = read_conll('./data/ner/dev.txt')

# 实例化trainer，设置参数，训练
trainer = BertCRFTrainer(
    pretrained_model_dir='../chinese-roberta-wwm-ext',
    model_dir='./bank_ner/tmp/bertcrf',
    learning_rate=5e-5,
    # 是否启用并行训练
    enable_parallel=True,
    # 设置focal_loss, 设置label权重分布(当前10个label，O的label_id为1)，需要降低O标签的权重
    loss_type='crf_loss',
    focal_loss_gamma=2,
    focal_loss_alpha=[1.] * 1 + [0.5] + [1.] * 8
)
trainer.train(
    train_texts,
    train_labels,
    validate_texts=dev_texts,
    validate_labels=dev_labels,
    batch_size=2,
    epochs=4
)

# 实例化predictor，加载模型，进行预测
predictor = BERTCRFPredictor(
    pretrained_model_dir='../chinese-roberta-wwm-ext',
    model_dir='./bank_ner/tmp/bertcrf/',
    # 是否启用并行预测
    enable_parallel=False
)
predict_labels = predictor.predict(
    dev_texts,
    batch_size=20
)

# 将结果输出为3列
out = open('./bank_ner/tmp/dev_results.txt', 'w', encoding='utf-8')
for text, each_true_lables, each_predict_labels in zip(dev_texts, dev_labels, predict_labels):
    for char, true_label, predict_label in zip(text, each_true_lables, each_predict_labels):
        out.write('{}\t{}\t{}\n'.format(char, true_label, predict_label))
    out.write('\n')
out.close()
# 进行宏f1评测
evaluate.eval('./bank_ner/tmp/dev_results.txt')