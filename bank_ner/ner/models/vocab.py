import json

from bank_ner.ner import logger


class Vocab:
    '''构建字或词及其id的映射、构建label及其id的映射'''
    def __init__(self, vocab2id=None, id2vocab=None, tag2id=None, id2tag=None,
                 unk_vocab_id=0, pad_vocab_id=1, pad_tag_id=0):
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab
        self.tag2id = tag2id
        self.id2tag = id2tag
        self.vocab_size = 0 if not vocab2id else len(vocab2id)
        self.label_size = 0 if not tag2id else len(tag2id)
        self.unk_vocab_id = unk_vocab_id
        self.pad_vocab_id = pad_vocab_id
        self.pad_tag_id = pad_tag_id

    def build_vocab(self, texts=None, labels=None, build_texts=True, build_labels=True, with_build_in_tag_id=True):
        '''构建词典'''
        logger.info('start to build vocab ...')
        if build_texts:
            assert texts, 'Please make sure texts is not None!'
            # 将unk、pad加入
            self.vocab2id, self.id2vocab = {'<UNK>': self.unk_vocab_id, '<PAD>': self.pad_vocab_id}, \
                                           {self.unk_vocab_id: '<UNK>', self.pad_vocab_id: '<PAD>'}
            # 初始vocab的计数为2，包含了unk跟pad的token
            vocab_cnt = 2
            for text in texts:
                # 迭代每一批文本
                for seg in text:
                    # 对于每一个文本里的字进行迭代
                    if seg in self.vocab2id:
                        # 如果该字已经在vocab2id中，则跳过
                        continue
                    # 将字添加到vocab2id中，id为vocab_cnt
                    self.vocab2id[seg] = vocab_cnt
                    self.id2vocab[vocab_cnt] = seg
                    # vocab_cnt自增
                    vocab_cnt += 1
            # 获得vocab的长度
            self.vocab_size = len(self.vocab2id)
        if build_labels:
            assert labels, 'Please make sure labels is not None!'
            # 将pad加入
            self.tag2id, self.id2tag = {'<PAD>': self.pad_tag_id}, {self.pad_tag_id: '<PAD>'}
            # 初始tag的计数为2，包含了pad的token
            tag_cnt = 1
            if not with_build_in_tag_id:
                # label不预置PAD_ID，则将tag2id、id2tag重置为空字典，tag_cnt计数重置为0
                self.tag2id, self.id2tag = {}, {}
                tag_cnt = 0
            for label in labels:
                # 迭代每一批label
                for each_label in label:
                    # 对于每批label的标签进行迭代
                    if each_label in self.tag2id:
                        # 如果该标签已经在tag2id中，则跳过
                        continue
                    # 将标签添加到tag2id中，id为tag_cnt
                    self.tag2id[each_label] = tag_cnt
                    self.id2tag[tag_cnt] = each_label
                    # tag_cnt自增
                    tag_cnt += 1
            # 获得tag的长度
            self.label_size = len(self.tag2id)
        logger.info('build vocab finish, vocab_size: {}, label_size: {}'.format(self.vocab_size, self.label_size))

    def save_vocab(self, vocab_file):
        '''保存词汇表'''
        result = {
            'vocab2id': self.vocab2id,
            'id2vocab': self.id2vocab,
            'tag2id': self.tag2id,
            'id2tag': self.id2tag
        }
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=4))
        logger.info('save vocab to {}'.format(vocab_file))

    def load_vocab(self, vocab_file):
        '''加载词汇表'''
        with open(vocab_file, 'r', encoding='utf-8') as f:
            result = json.loads(f.read())
        self.vocab2id = result['vocab2id']
        # 将id转为int类型
        self.id2vocab = {int(k): v for k, v in result['id2vocab'].items()}
        self.tag2id = result['tag2id']
        # 将id转为int类型
        self.id2tag = {int(k): v for k, v in result['id2tag'].items()}
        self.vocab_size = len(self.vocab2id)
        self.label_size = len(self.tag2id)

    '''
    当使用预训练语言模型或embedding时，有可能使用别人的vocab，下面的set方法允许设置外部的vocab
    '''
    def set_vocab2id(self, vocab2id):
        self.vocab2id = vocab2id
        self.vocab_size = len(self.vocab2id)
        return self

    def set_id2vocab(self, id2vocab):
        self.id2vocab = id2vocab
        return self

    def set_tag2id(self, tag2id):
        self.tag2id = tag2id
        self.label_size = len(self.tag2id)
        return self

    def set_id2tag(self, id2tag):
        self.id2tag = id2tag
        return self

    def set_unk_vocab_id(self, unk_vocab_id):
        self.unk_vocab_id = unk_vocab_id

    def set_pad_vocab_id(self, pad_vocab_id):
        self.pad_vocab_id = pad_vocab_id

    def set_pad_tag_id(self, pad_tag_id):
        self.pad_tag_id = pad_tag_id