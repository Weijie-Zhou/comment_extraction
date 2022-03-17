import json
from json.tool import main

import pandas as pd
from sklearn.model_selection import train_test_split


def save_sentiment_data(reviews, labels, save_path):
    # 将数据保存为json格式
    with open(save_path, 'w', encoding='utf8') as f:
        for review, label in zip(reviews, labels):
            dic = {}
            dic['inputs'] = review
            dic['labels'] = label
            f.write(json.dumps(dic, ensure_ascii=False))
            f.write('\n')


def data_sentiment_preprocess(path):
    df = pd.read_csv(path, encoding='utf8')
    sentiment_labels_map = {'负面':0, '正面':1, '中立': 2}
    sentiment_labels_id = {int(value): key for key, value in sentiment_labels_map.items()}
    df['class_text'] = df['class'].apply(lambda x: sentiment_labels_id[x])
    X_train, X_test, y_train, y_test = train_test_split(
                                df['text'], df['class_text'], test_size=0.15, random_state=1024)
    print('X_train:', len(X_train), 'X_test:', len(X_test))
    save_sentiment_data(X_train, y_train, './data/sentiment/train.json')
    save_sentiment_data(X_test, y_test, './data/sentiment/eval.json')
    # 保存config文件
    config_dic = {}
    config_dic['train_size'] = len(X_train)
    config_dic['eval_size'] = len(X_test)
    config_dic['task_tag'] = 'classification'
    config_dic['labels_map'] = sentiment_labels_map
    config_dic['labels_balance'] = {
        'train_labels': {'负面': sum(y_train == '负面'), '正面': sum(y_train == '正面'), '中立': sum(y_train == '中立')},
        'eval_labels': {'负面': sum(y_test == '负面'), '正面': sum(y_test == '正面'), '中立': sum(y_test == '中立')},
    }
    with open('./data/sentiment/config.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(config_dic, ensure_ascii=False))
    print('sentiment data preprocess finish!')


def save_ner_data(x, y, save_path):
    # 将数据保存为json格式
    with open(save_path, 'w', encoding='utf8') as f:
        for text, labels in zip(x, y):
            assert len(text) == len(labels.split()), 'text length must be equal to labels length'
            for char, label in zip(text, labels.split()):
                f.write('{}\t{}\n'.format(char, label))
            f.write('\n')


def data_ner_preprocess(train_path, test_path):
    df = pd.read_csv(train_path, encoding='utf8')
    # 切分训练集、验证集
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['BIO_anno'], test_size=0.15, random_state=1024)
    print('X_train:', len(X_train), 'X_test:', len(X_test))
    # 保存训练集与验证集
    save_ner_data(X_train, y_train, 'data/ner/train.txt')
    save_ner_data(X_test, y_test, 'data/ner/dev.txt')
    # 对测试集进行处理
    test_df = pd.read_csv(test_path, encoding='utf8')
    with open('data/ner/test.txt', 'w', encoding='utf8') as f:
        for test_text in test_df['text']:
            for char in test_text:
                f.write('{}\n'.format(char))
            f.write('\n')
    print('bank_ner data preprocess finish!')


if __name__ == '__main__':
    train_path = './data/train_data_public.csv'
    test_path = './data/test_public.csv'
    data_sentiment_preprocess(train_path)
    data_ner_preprocess(train_path, test_path)