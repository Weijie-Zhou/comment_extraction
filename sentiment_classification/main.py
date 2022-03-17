import math
import os
import random
import argparse
import numpy as np
import logging
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from Dataset import DataIterator
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score
from config import Config
from bert_classification_model import BertClassificationModel

seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--train_epochs', type=int, default=4)
parser.add_argument('--eval_epochs', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=100)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--corpus_tag_or_dir', type=str, default='./data/sentiment')
parser.add_argument('--pretrain_model_path', type=str, default='../chinese-roberta-wwm-ext')
parser.add_argument('--num_labels', type=int, default=3)
parser.add_argument('--model_save_path', type=str, default='./sentiment_classification/output')
parser.add_argument('--log_save_path', type=str, default='./sentiment_classification/log')
args = parser.parse_args()
con = Config(args)


# 初始化设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始数据集
data = DataIterator.from_corpus(
    corpus_tag_or_dir=con.corpus_tag_or_dir,
    tokenizer_path=con.pretrain_model_path,
    batch_size=con.batch_size,
    shuffle=False, # DataItrator的shuffle需要False
    max_length=con.max_len,
    device=device
)
# 获得训练集迭代器、验证集迭代器、配置config参数
train_iter, eval_iter, config = data['train_iter'], data['eval_iter'], data['config']

# 初始化模型
model = BertClassificationModel(
    bert_base_model_dir=con.pretrain_model_path,
    label_size=con.num_labels,
    device=device,
    loss_type='focal_loss',
    focal_loss_gamma=2,
    focal_loss_alpha= [1.0] + [1.0] + [0.5]
)
# 将模型设置为训练模式
model.train()
# 将模型拷贝到设备上
model.to(device)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=con.learning_rate, weight_decay=0.01)
# 初始化混合精度训练的训练器，平常的参数是float32: 4bytes，效率较慢，
# 而混合精度在前向传播时使用float16: 2bytes，提升效率
grad_scaler = GradScaler()

# 初始化保存路径
# 设置日志打印等级
logging.getLogger().setLevel(logging.INFO)
# 初始化tensorboardX日志加载器
summaries_writer = SummaryWriter(con.log_save_path)
# 模型保存路径
model_save_path = con.model_save_path

# 初始化训练参数
total_train_epochs = con.train_epochs # 总迭代轮数
total_eval_epochs = con.eval_epochs # 验证轮数
eval_per_n_epochs = con.eval_epochs # 每训练几轮，进行一次验证
log_per_n_steps = con.log_steps # 每训练多少步，进行一次日志打印


def train_with_eval():
    # 计算迭代参数
    # 总训练步数
    total_train_steps = math.ceil(config['train_size'] * total_train_epochs / config['batch_size'])
    print('total_train_steps:', total_train_steps)
    # 总验证步数
    total_eval_steps = math.ceil(config['eval_size'] * total_eval_epochs / config['batch_size'])
    print('total_eval_steps:', total_eval_steps)
    # 每隔多少步进行一次验证
    eval_per_n_steps = math.ceil(config['train_size'] * eval_per_n_epochs / config['batch_size'])
    # 每个epoch有多少步
    steps_one_epoch = math.ceil(config['train_size'] / config['batch_size'])
    # 初始化最优f1
    best_f1 = -float('inf')
    # 训练
    for train_step in range(total_train_steps):
        if train_step % steps_one_epoch == 0:
            # 打印当前是第几轮epoch
            logging.info(f'start {int(train_step / steps_one_epoch) + 1}th training epoch')
        elif train_step > 0 and train_step % log_per_n_steps == 0:
            # 每隔log_per_n_steps打印一次日志
            logging.info(f'progress to {train_step}th training step')
        # 通过next得到one_batch数据
        train_data = next(train_iter)
        # 前向forward，得到loss
        with autocast():
            # 通过with autocast开启前向混合精度训练
            train_loss, _ = model(**train_data)
        logging.info(f'{int(train_step / steps_one_epoch + 1)}th epoch {train_step}th step training loss is {train_loss}')
        # 通过summaries_writer记录loss以及当前训练步数(train_loss为纵轴的值，train_step为横轴的值)
        summaries_writer.add_scalar('train_loss', train_loss, train_step)
        # 反向传播backward，更新参数
        optimizer.zero_grad() # 梯度清零
        # 在开启了混合精度训练以及梯度缩放后，通过以下方式进行参数更新
        grad_scaler.scale(train_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        # 进行测试
        if eval_iter is not None:
            if train_step > 0 and train_step % eval_per_n_steps == 0 or train_step == total_train_steps-1:
                logging.info(f'start {int(train_step / eval_per_n_steps)}th model evaluation')
                # 将模型设置为评估模式
                model.eval()
                # 保存预测结果
                predictions = []
                # 保存实际标签
                labels = []
                for eval_step in range(total_eval_steps):
                    # 通过next得到one_batch数据
                    eval_data = next(eval_iter)
                    # 提取真实label标签
                    batch_labels = eval_data['labels']
                    # 将真实标签加入到labels中
                    labels += batch_labels.detach().cpu().numpy().tolist()
                    # 前向forward
                    eval_loss, logits = model(**eval_data)
                    # 通过argmax提取概率最大值的索引来获得预测标签的id
                    batch_predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
                    # 将预测结果加入到predictions
                    predictions += batch_predictions
                # 计算weighted的f1
                f1 = f1_score(labels, predictions, average='weighted')
                logging.info(f'{int(train_step / eval_per_n_steps)}th epoch f1 score is {f1}')
                # 将f1记录到summaries_writer
                summaries_writer.add_scalar('eval f1', f1, int(train_step / eval_per_n_steps))
                logging.info(f'{int(train_step / eval_per_n_steps)}th model evaluation is completed')
                # 如果当前验证集的f1比best_f1要大，则进行模型保存，并更新best_f1
                if f1 > best_f1:
                    best_f1 = f1
                    logging.info(f'start {int(train_step / eval_per_n_steps)}th model saving')
                    if not os.path.exists(model_save_path):
                        os.mkdir(model_save_path)
                    model_save_name = os.path.join(model_save_path, 'pytorch_model.h5')
                    torch.save(model, model_save_name)
                    logging.info(f"{int(train_step / eval_per_n_steps)}th model saving is completed.")
                # 将模型重新设置为训练模式
                model.train()
        if (train_step + 1) % steps_one_epoch == 0:
            logging.info(f"{int(train_step / steps_one_epoch) + 1}th training epoch is completed.")
    return


if __name__ == '__main__':
    train_with_eval()