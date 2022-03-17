import json
import math
import os
from abc import ABC
from typing import Iterator, Optional, Dict
import torch
from torch.utils import data
from torch.utils.data import IterableDataset, TensorDataset
from torch.utils.data.dataset import T_co
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def yield_from(iter_list):
    # 接收一个迭代器列表，然后通过yield from方式，将多个迭代器列表合并成一个迭代器列表
    for it in iter_list:
        yield from it


class FilesDataset(IterableDataset, ABC):
    # IterableDataset：pytorch中一个待实现的类，不能被直接调用，需要自己封装这个类
    # 该方法接收多个文件名，将这些文件依次打开，合并成一个迭代器
    # 同时继承ABC的原因是由于IterableDataset处于__init__以及__iter__方法之外还有其他方法要实现，
    # 但是我们在这里不实现，因此为了不报错，警告，我们让FilesDataset同时继承ABC类，把那些待实现的方法遮住
    def __init__(self, filenames):
        # 接收一个文件名列表[A, B, C]，或者一个单词的字符串的文件名A
        super().__init__()
        if isinstance(filenames, str):
            # A -> [A]，将字符串的文件名A，转成列表形式
            filenames =  [filenames]
        self.filenames = filenames

    def __iter__(self) -> Iterator[T_co]:
        # 实现内部迭代的方法，这是要实现IterableDataset类时必须要实现的方法
        # 作用：当我们实现的DataLoader类去调用IterableDataset时，他的内部会使用这样的方式调用数据
        # 我们需要告诉IterableDataset，我们要如何切分各种各样的任务(假如我们有3个文件，就有3个任务)
        worker_info = torch.utils.data.get_worker_info() # 得到pytorch内核类的信息
        if worker_info is None:
            # 如果worker_info是None，这代表我们是以单进程的方式启动Dataset，那么会把所有的文件都通过open方式打开
            # 打开之后，每个文件都是一个迭代器，我们可以通过yield_from方法把多个迭代器合并成同一个迭代器
            generator = yield_from([open(fn, encoding='utf-8') for fn in self.filenames])
        else:
            # 如果worker_info不为None，则说明是以多进程方式启动Dataset
            # 因此我们需要获知当前被启动的数据集是第几个worker，例如我们可能以多进程的方式读取10个文件
            # 假设我们启动了5个进程，那么pytorch内部是启动了5个实例，每个实例内部都是一个Dataset
            # 而每个实例里的Dataset都有一个id，我们根据id划分10个任务，然后每个id会得到2个任务
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # 我们通过有多少个任务/多少个worker的方式，计算出每个worker要有多少个任务
            per_worker = int(math.ceil(len(self.filenames) / float(num_workers)))
            # 计算每个worker的起始任务点和终止任务点
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.filenames))
            # 根据起始、终止点，将对应的任务文件依次打开，打开之后由yield_from方法，将打开的多个任务文件迭代器合并成同一个迭代器
            generator = yield_from([open(fn, encoding='utf-8') for fn in self.filenames[iter_start: iter_end]])
        return generator


class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        '''
        :param dataset: IterableDataset实例化后的dataset
        :param batch_size: 批次大小
        :param shuffle: 是否shuffle
        :param num_workers: 控制单进程还是多进程
        :param kwargs: 其他可选参数
        '''
        # 调用父类方法进行初始化
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    @classmethod
    def from_tensor(cls, tensors, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        # 从tensor中实现一个dataloader，加载数据集。这是类方法，在DataLoader实例化之前就可以调用(可以绕过实例需要初始化参数)
        # tensors：可以是二维列表，也可以二维的array、tensor等等
        for i in range(len(tensors)):
            # 将传入的tensors转成torch的tensor格式
            tensor = tensors[i]
            if not isinstance(tensor, torch.Tensor):
                tensors[i] = torch.Tensor(tensor)
        # 使用TensorDataset对tensors进行加载，TensorDataset与IterableDataset是平行类，专门用来加载内存中的数据集
        dataset = TensorDataset(*tensors)
        # 将加载好的dataset传给初始化的方法，得到一个DataLoader实例(cls方法解决了必须要实例化一个类，才能调用它的方法的悖论)
        return cls(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    @classmethod
    def from_file(cls, filenames, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        # 从文件中实现一个dataloader，加载数据集。这是类方法，在DataLoader实例化之前就可以调用(可以绕过实例需要初始化参数)
        # filenames: 文件名列表或字符串文件名
        if isinstance(filenames, str):
            # 如果filenames是一个字符串，则将其转成列表格式
            filenames = [filenames]
        # 使用FilesDataset对文件进行加载
        dataset = FilesDataset(filenames)
        # 将加载好的dataset传给初始化的方法，得到一个DataLoader实例(cls方法解决了必须要实例化一个类，才能调用它的方法的悖论)
        return cls(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)


def map_func_basic(one_batch, tokenizer, is_split_into_words, labels_map, max_length, preprocess_fn):
    # 基础数据转换函数
    # in: one batch = [{'inputs': 'xxx', 'labels': 'yy'}, {...}, ...]
    # out: one batch = {'inputs': ['xxx', 'xxx2', ...], 'labels': ['yy', 'yy2', ...]}
    # 得到batch_size
    batch_size = len(one_batch)
    # 将one_batch的数据使用json解码，并添加到列表中
    one_batch = [json.loads(item.strip()) for item in one_batch]
    # 获得one_batch数据的键
    keys = one_batch[0].keys()
    # 初始化缓存batch
    cache_batch = {}
    # 判断inputs是否包含在keys中
    assert 'inputs' in keys, 'corpus must contain key: inputs.'
    if 'labels' in keys:
        # 如果labels在keys中，则需要判断labels_map标签字典是否存在
        assert labels_map is not None, 'if corpus has labels, labels_map can not be None'
    # 遍历所有sample，将按样本区分的one batch转为按key区分的one batch
    for item in one_batch:
        for key in keys:
            if key == 'labels':
                # 使用labels_map编码labels，将label转换成该标签对应的id
                encoded_label = labels_map[item[key]]
                # 将转换好的label保存到cache_batch中
                cache_batch.setdefault(key, []).append(encoded_label)
            elif key == 'inputs':
                # 获得inputs数据
                preprocessed = item[key]
                if preprocess_fn is not None:
                    # 如果preprocess_fn存在，则对inputs数据进行预处理，例如小写化，标点符号统一，全半角转换等
                    preprocessed = preprocess_fn(item[key])
                # 将inputs数据保存到cache_batch中
                cache_batch.setdefault(key, []).append(preprocessed)
    # 编码inputs：{'input_ids': tensor, 'token_type_ids': tensor, 'attention_mask': tensor}
    # 使用分词器的batch_encode_plus进行编码
    # is_split_to_words：输入是否已预标记(例如，拆分为单词),如果为True，在这种情况下，标记器将跳过预标记化步骤。这对于NER或令牌分类很有用。
    encoded = tokenizer.batch_encode_plus(batch_text_or_text_pairs=cache_batch['inputs'],
                                          padding=True,
                                          is_split_into_words=is_split_into_words,
                                          max_length=max_length,
                                          truncation=True)
    # 组装成新的one batch
    new_batch = {}
    if 'labels' in keys:
        new_batch.update({'labels': cache_batch['labels']})
    for key, value in encoded.items():
        new_batch[key] = value
    # 转为tensor
    for key in new_batch.keys():
        new_batch[key] = torch.tensor(new_batch[key])
    return new_batch



class DataIterator:
    def __init__(self, data_loader, map_func=None, device=None):
        '''
        :param data_loader: DataLoader的实例
        :param map_func: 数据从dataloader中取出到，DataIterator返回之间的数据转换函数
        :param device: 设备
        '''
        self.data_loader = data_loader # 将data_loader保存到实例变量中
        self.iterator = iter(data_loader) # 使用iter方法对data_loader进行调用，得到iterator迭代器
        self.map_func = map_func # 将map_func保存到实例变量中
        # 定义设备，适配各种传参格式
        if isinstance(device, torch.device):
            self.device = device
        elif type(device) == str:
            self.device = torch.device(device)
        elif device==None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            raise TypeError("device must be str or instance of torch.device.")

    @classmethod
    def from_tensor(cls, tensors, batch_size=1, shuffle=False, num_workers=0, map_func=None, device=None):
        # 从内存中获取数据集
        # 调用DataLoader.from_tensor类方法，加载内存中的数据集，进而得到一个data_loader
        data_loader = DataLoader.from_tensor(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        # 调用DataIterator的初始化方法，将data_loader传入，实例化一个data_iterator，返回
        return cls(data_loader, map_func, device)

    @classmethod
    def from_file(cls, filenames, batch_size=1, shuffle=False, num_workers=0, map_func=None, device=None):
        # 从内存中获取数据集
        # 调用DataLoader.from_file类方法，加载文件中的数据集，进而得到一个data_loader
        data_loader = DataLoader.from_file(filenames, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        # 调用DataIterator的初始化方法，将data_loader传入，实例化一个data_iterator，返回
        return cls(data_loader, map_func, device)

    @classmethod
    def from_corpus(cls,
                    corpus_tag_or_dir: str, # 标准数据集的文件夹路径
                    tokenizer_path: str, # 分词器的路径
                    batch_size: int=1, # 批次大小
                    shuffle: bool=False, # 是否对数据乱序
                    num_workers: int=0, # 设置多少个进程
                    is_split_into_words: Optional[bool]=False, # 在bert调用tokenizer时传入
                    labels_map: Optional[Dict[str, int]]=None, # 标签的字典
                    max_length: int=512, # 输入序列最大长度
                    preprocess_fn=None, # 数据预处理函数
                    device=None # 设备
                    ):
        # from_corpus是对from_file方法的标准化，当要加载标准数据集的时候，那么就可以不用重新写map_func，只需要提供一些额外的参数即可
        # 得到数据集路径和config
        train_path = os.path.join(corpus_tag_or_dir, 'train.json') # 训练集路径
        eval_path = os.path.join(corpus_tag_or_dir, 'eval.json') # 验证集路径
        config = json.load(open(os.path.join(corpus_tag_or_dir, 'config.json'))) # 加载config文件，得到相关参数
        # 将batch_size、shuffle、num_workers保存到config中
        config['batch_size'] = batch_size
        config['shuffle'] = shuffle
        config['num_workers'] = num_workers
        # 实例化data_loader，通过DataLoader的from_file类方法，加载训练集data_loader
        train_loader = DataLoader.from_file(train_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if os.path.exists(eval_path):
            # 如果存在验证集路径，则实例化验证集data_loader
            eval_loader = DataLoader.from_file(eval_path, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            # 如果没有验证集路径，eval_loader为None
            eval_loader = None
        # 初始化tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 加载labels_map 标签字典
        if labels_map is None:
            # 如果labels_map没有传入，则从config文件里获得labels_map
            labels_map = config.get('labels_map', None)

        def map_func(one_batch):
            # 定义map_func，传入one_batch数据集，输出一个经过map_func转换后的one_batch数据集
            return map_func_basic(one_batch, tokenizer, is_split_into_words, labels_map, max_length, preprocess_fn)

        # 生成iterator，对训练集进行实例化，得到训练集的迭代器iterator，同时将config也传入
        data_iter = {'train_iter': cls(train_loader, map_func, device), 'config': config}
        if eval_loader is not None:
            # 如果验证集存在，则对验证集进行实例化，将得到的验证集的迭代器iterator加入data_iter字典中
            data_iter.update({'eval_iter': cls(eval_loader, map_func, device)})
        return data_iter

    def __next__(self):
        # batch = next(data_iterator)
        # one_batch: [{key1: '', key2: ''}, {key1: '', key2: ''}, ...] => {key1: [], key2: [], ...}
        # 控制data_iterator在next的一些行为，我们在使用next方法后，会得到一些batch数据，需要对batch数据进行一些转换，得到最终的数据
        try:
            # 通过next方法，调用self.iterator，得到one_batch数据
            one_batch = next(self.iterator)
        except StopIteration:
            # 为了解决next方法在调用的时候，在所有数据迭代完后会出现的StopIteration报错，由此解决了有限次调用的问题，进而可以任意调用数据集
            # 我们对self.iterator，重新使用iter对data_loader进行调用，然后再使用next进行调用，获得one_batch数据
            self.iterator = iter(self.data_loader)
            one_batch = next(self.iterator)
        if self.map_func is not None:
            # 如果map_func不为None，则对one_batch数据进行转换
            one_batch = self.map_func(one_batch)
        for k in one_batch:
            # 将one_batch里的数据拷贝到设备上
            one_batch[k] = one_batch[k].to(self.device)
        return one_batch

    def __iter__(self):
        # iter(data_iterator) => self
        # 如果使用iter(data_iterator)，得到就是他自己，通过这样的方式解决一些误操作，因为iter方法应该使用在data_loader上
        return self