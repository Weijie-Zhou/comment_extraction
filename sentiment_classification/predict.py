import math
import time
import json

import pandas as pd
import torch
from transformers import AutoTokenizer


# 初始化设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('../chinese-roberta-wwm-ext')
# 加载模型
model = torch.load('./sentiment_classification/output/pytorch_model.h5', map_location=device)
model.eval()
# 将模型拷贝到设备上
model.to(device)
# 加载labels_map
labels_map = json.load(open('./data/sentiment/config.json'))['labels_map']
labels_map_id = {int(value): key for key, value in labels_map.items()}

def predict_one(review):
    # 对一条传入的样本进行预测
    if type(review) == str:
        review = [review]
    encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=review,
                                          padding=True,
                                          max_length=128,
                                          truncation=True)
    new_batch = {}
    for key, value in encoded.items():
        new_batch[key] = torch.tensor(value)
    eval_output = model(**new_batch)
    # 通过argmax提取概率最大值的索引来获得预测标签的id
    batch_predictions = torch.argmax(eval_output.logits, dim=-1).detach().cpu().numpy().tolist()
    # 将预测结果加入到predictions
    return labels_map_id[batch_predictions[0]]

def predict_batch(reviews, batch_size):
    # 对传入的样本按照batch_size大小进行批预测
    predict_steps = math.ceil(len(reviews) / batch_size)
    print('predict_steps: ', predict_steps)
    # 保存预测结果
    predictions = []
    for i in range(predict_steps):
        print('now step: ', i)
        review_lst = list(reviews[i*batch_size: (i+1)*batch_size])
        encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=review_lst,
                                              padding=True,
                                              max_length=128,
                                              truncation=True)
        new_batch = {}
        for key, value in encoded.items():
            new_batch[key] = torch.tensor(value, device=device)
        eval_loss, logits = model(**new_batch)
        # 通过argmax提取概率最大值的索引来获得预测标签的id
        batch_predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        # 将预测结果加入到predictions
        predictions += [labels_map_id[prediction] for prediction in batch_predictions]
    return predictions


if __name__ == '__main__':
    start = time.time()
    test_data = []
    with open('data/test_public.csv', 'r', encoding='utf8') as f:
        for line in f.readlines()[1:]:
            if line.strip() == '':
                continue
            test_data.append(line.strip())
    print(len(test_data))
    predict_labels = predict_batch(test_data, 32)
    print(len(predict_labels), len(test_data))
    print(time.time() - start)
    with open('submission_sentiment.csv', 'w', encoding='utf8') as f:
        f.write('id,class\n')
        for index, label in enumerate(predict_labels):
            f.write('{},{}\n'.format(index, labels_map[label]))

    # 合并ner以及情感分类的数据
    ner_data = pd.read_csv('submission_ner.csv')
    sentiment_data = pd.read_csv('submission_sentiment.csv')
    submission = pd.concat([ner_data, sentiment_data['class']], axis=1)
    submission.to_csv('submission.csv', index=False)