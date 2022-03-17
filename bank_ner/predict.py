from bank_ner.ner.models.bert_crf.bert_crf_predictor import BERTCRFPredictor


def read_conll(file_path):
    # 读取数据
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts = []
        for example in f.read().split('\n\n'):
            # 迭代每条样本
            example = example.strip()
            if not example:
                continue
            texts.append([])
            for term in example.split('\n'):
                if len(term.strip()) != 1:
                    # 跳过不合法的行
                    continue
                texts[-1].append(term)
        return texts


def predict_labels(texts):
    predictor = BERTCRFPredictor(
        pretrained_model_dir='../chinese-roberta-wwm-ext',
        model_dir='./bank_ner/tmp/bertcrf/',
        # 启用并行预测
        enable_parallel=False
    )
    predict_labels = predictor.predict(
        texts,
        batch_size=64
    )
    return predict_labels



if __name__ == '__main__':
    test_path = './data/ner/test.txt'
    test_texts = read_conll(test_path)
    # test_texts = [['共', '享', '一', '个', '额', '度', '，', '没', '啥', '必', '要', '，', '四', '个', '卡', '不', '要', '年', '费', '吗', '？', '你', '这', '种', '人', '头', '，', '银', '行', '最', '喜', '欢', '，', '广', '发', '是', '出', '了', '名', '的', '风', '控', '严', '，', '套', '现', '就', '给', '你', '封', '.', '.', '.']]
    labels_batch = predict_labels(test_texts)

    with open('submission_ner.csv', 'w', encoding='utf8') as f:
        f.write('id,BIO_anno\n')
        for index, labels in enumerate(labels_batch):
            assert len(labels) == len(test_texts[index]), 'predict labels length must be euqal to origin texts'
            f.write('{},{}\n'.format(index, ' '.join(labels)))