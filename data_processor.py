import json
import random
import re

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
from torchtext.vocab import build_vocab_from_iterator, Vectors, vocab

"""
因为数据集比较大，这里一次处理完数据保存好直接读取
主要数据包括引文网络（节点属性是用固定的还是根据模型训练变动需要考虑，最基本可能要包括文本的编号序列）、预测序列（暂时是简单的窗口预测）
BERT因为比较大，可能只能用来获得当前文章的rep，引文网络上其他节点的特征只能预先训练转化好、或减少bs、或筛选固定ref、防止爆显存
"""

# tokenizer = get_tokenizer('basic_english')
UNK, PAD, SEP = '[UNK]', '[PAD]', '[SEP]'


class DataProcessor:
    def __init__(self, data_source, max_len=256, use_graph=False, seed=123):
        print('Init...')
        self.data_root = './data/'
        self.data_source = data_source
        self.seed = int(seed)
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        if self.data_source == 'pubmed':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 's2orc':
            self.data_cat_path = self.data_root + self.data_source + '/'
        elif self.data_source == 'pubmed':
            self.data_cat_path = self.data_root + self.data_source + '/'

    def split_data(self, rate=0.8, fixed_num=None, shuffle=True, by='normal'):
        # 这里注意把这步之前的数据都处理好，info_dict中只保留必要的内容，同时处理成统一格式
        all_values = json.load(open(self.data_cat_path + 'sample_citation_accum.json'))
        info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json'))
        print(len(all_values))
        all_ids = list(all_values.keys())

        if by == 'time':
            print('underwork')
            raise Exception
        else:
            if shuffle:
                random.seed(self.seed)
                print('data_processor seed', self.seed)
                random.shuffle(all_ids)

        total_count = len(all_ids)
        train_ids = all_ids[:int(total_count * rate)]
        val_ids = all_ids[int(total_count * rate): int(total_count * ((1 - rate) / 2 + rate))]
        test_ids = all_ids[int(total_count * ((1 - rate) / 2 + rate)):]

        train_values = list(map(lambda x: all_values[x], train_ids))
        val_values = list(map(lambda x: all_values[x], val_ids))
        test_values = list(map(lambda x: all_values[x], test_ids))

        # train_contents = list(map(lambda x: ' '.join(tokenizer(info_dict[x]['title'] + '. ' + info_dict[x]['abstract'])), train_ids))
        train_contents = list(
            map(lambda x: re.sub('\s', ' ', info_dict[x]['title'] + '. ' + info_dict[x]['abstract']), train_ids))
        val_contents = list(
            map(lambda x: re.sub('\s', ' ', info_dict[x]['title'] + '. ' + info_dict[x]['abstract']), val_ids))
        test_contents = list(
            map(lambda x: re.sub('\s', ' ', info_dict[x]['title'] + '. ' + info_dict[x]['abstract']), test_ids))

        cut_data = {
            'train': [train_ids, train_values, train_contents],
            'val': [val_ids, val_values, val_contents],
            'test': [test_ids, test_values, test_contents],
        }

        torch.save(cut_data, self.data_cat_path + 'split_data')

    def get_tokenizer(self, type='basic', tokenizer_path=None):
        if type == 'bert':
            self.tokenizer = CustomBertTokenizer(max_len=self.max_len, bert_path=tokenizer_path,
                                                 data_path=self.data_cat_path)
        elif type == 'glove':
            self.tokenizer = VectorTokenizer(max_len=self.max_len, vector_path=tokenizer_path,
                                             data_path=self.data_cat_path, name='glove')
        else:
            self.tokenizer = BasicTokenizer(max_len=self.max_len, data_path=self.data_cat_path)

        data = torch.load(self.data_cat_path + 'split_data')
        self.tokenizer.build_vocab(data['train'][2], seed=self.seed)

    def get_dataloader(self, batch_size=32, num_workers=0):
        data = torch.load(self.data_cat_path + 'split_data')
        # all_last_values = map(lambda x: x[1][-1], data['train'][1])
        train_dataloader = DataLoader(dataset=list(zip(*data['train'])), batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers, collate_fn=self.collate_batch)
        val_dataloader = DataLoader(dataset=list(zip(*data['val'])), batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, collate_fn=self.collate_batch)
        test_dataloader = DataLoader(dataset=list(zip(*data['test'])), batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers, collate_fn=self.collate_batch)
        self.dataloaders = [train_dataloader, val_dataloader, test_dataloader]
        return self.dataloaders

    def get_feature_graph(self, tokenizer_path, mode='vector'):
        graph = torch.load(self.data_cat_path + 'graph_sample')
        info_dict = json.load(open(self.data_cat_path + 'sample_info_dict.json', 'r'))
        node_trans = json.load(open(self.data_cat_path + 'sample_node_trans.json', 'r'))
        node_ids = list(node_trans.values())
        node_ids.sort()
        print(len(node_ids))
        index_trans = dict(zip(node_trans.values(), node_trans.keys()))
        abstracts = list(map(lambda x: info_dict[index_trans[x]]['abstract'], node_ids))
        feature_list = []
        del info_dict

        if mode == 'vector':
            self.tokenizer = VectorTokenizer(max_len=self.max_len, vector_path=tokenizer_path,
                                             data_path=self.data_cat_path, name='glove')
            self.tokenizer.load_vocab()
            print(self.tokenizer.vectors.shape)

            for abstract in abstracts:
                processed_content, seq_len, mask = self.tokenizer.encode(abstract)
                node_embedding = self.tokenizer.vectors[processed_content][:seq_len].mean(dim=0, keepdim=True)
                feature_list.append(node_embedding)

            graph.ndata['h'] = torch.cat(feature_list, dim=0)
            print(graph.ndata['h'].shape)

        torch.save(graph, self.data_cat_path + 'graph_sample_feature')


    @staticmethod
    def values_pipeline(values):
        inputs = values[0]
        outputs = values[1]
        inputs = list(filter(lambda x: x >= 0, inputs))
        valid_len = len(inputs)
        inputs = inputs + [0] * (len(outputs) - valid_len)
        return inputs, valid_len, outputs

    def collate_batch(self, batch):
        values_list, content_list = [], []
        inputs_list, valid_lens = [], []
        length_list = []
        mask_list = []
        ids_list = []
        # print(batch)
        for (_ids, _values, _contents) in batch:
            # processed_content, seq_len, mask = self.text_pipeline(_content)
            # # print(_label)
            processed_content, seq_len, mask = self.tokenizer.encode(_contents)
            # values_list.append(_values)
            inputs, valid_len, outputs = self.values_pipeline(_values)
            # values_list.append(self.label_pipeline(_values))
            values_list.append(_values)
            inputs_list.append(inputs)
            valid_lens.append(valid_len)
            content_list.append(processed_content)
            length_list.append(seq_len)
            mask_list.append(mask)
            # ids_list.append(int(_ids.strip()))
            ids_list.append(_ids.strip())
        # content_list = torch.cat(content_list)
        # 固定长度转换为张量
        content_batch = torch.tensor(content_list, dtype=torch.int64)
        values_list = torch.tensor(values_list, dtype=torch.int64)
        inputs_list = torch.tensor(inputs_list, dtype=torch.int64)
        valid_lens = torch.tensor(valid_lens, dtype=torch.int8)
        length_list = torch.tensor(length_list, dtype=torch.int64)
        mask_list = torch.tensor(mask_list, dtype=torch.int8)
        # ids_list = torch.tensor(ids_list, dtype=torch.int64)
        # print(len(label_list))
        content_list = [content_batch, inputs_list, valid_lens]
        return content_list, values_list, length_list, \
               mask_list, ids_list


class BasicTokenizer:
    def __init__(self, max_len=256, data_path=None):
        self.tokenizer = get_tokenizer('basic_english')
        self.max_len = max_len
        self.vocab = None
        self.data_path = data_path

    def yield_tokens(self, data_iter):
        # 转换成生成器形式，以便后续进行处理
        for content in data_iter:
            yield self.tokenizer(content)

    def build_vocab(self, text_list, seed=None):
        self.vocab = build_vocab_from_iterator(self.yield_tokens(text_list), specials=[UNK, PAD])
        self.vocab.set_default_index(self.vocab[UNK])
        torch.save(self.vocab, self.data_path + 'vocab_{}'.format(seed))

    def load_vocab(self, seed):
        self.vocab = torch.load(self.data_path + 'vocab_{}'.format(seed))

    def encode(self, text):
        tokens = self.tokenizer(text)
        seq_len = len(tokens)
        if seq_len <= self.max_len:
            tokens += (self.max_len - seq_len) * [PAD]
        else:
            tokens = tokens[:self.max_len]
            seq_len = self.max_len
        ids = self.vocab(tokens)
        masks = [1] * seq_len + [0] * (self.max_len - seq_len)
        return ids, seq_len, masks


class CustomBertTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, bert_path=None, data_path=None):
        super(CustomBertTokenizer, self).__init__(max_len)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def build_vocab(self, text_list, seed=None):
        self.vocab = {
            PAD: self.tokenizer.convert_tokens_to_ids([PAD])[0],
            UNK: self.tokenizer.convert_tokens_to_ids([UNK])[0],
            SEP: self.tokenizer.convert_tokens_to_ids([SEP])[0]
        }
        print('bert already have vocab')

    def load_vocab(self, seed=None):
        print('bert already have vocab')

    def encode(self, text):
        result = self.tokenizer(text)
        result = self.tokenizer.pad(result, padding='max_length', max_length=self.max_len)
        ids = result['input_ids']
        mask = result['attention_mask']
        seq_len = sum(mask)
        # SEP_IDX = self.tokenizer.convert_tokens_to_ids([SEP])
        SEP_IDX = self.tokenizer.vocab[SEP]
        if seq_len > self.max_len:
            ids = ids[:self.max_len - 1] + [SEP_IDX]
            mask = mask[:self.max_len]
            seq_len = self.max_len
        return ids, seq_len, mask


class VectorTokenizer(BasicTokenizer):
    def __init__(self, max_len=256, vector_path=None, data_path=None, name=None):
        super(VectorTokenizer, self).__init__(max_len, data_path)
        self.vector_path = vector_path
        self.vectors = None
        self.name = name

    def build_vocab(self, text_list=None, seed=None):
        vec = Vectors(self.vector_path)
        self.vocab = vocab(vec.stoi, min_freq=0)  # 这里的转换把index当成了freq，为保证一一对应设置为0，实际上不影响后续操作
        self.vocab.append_token(UNK)
        self.vocab.append_token(PAD)
        self.vocab.set_default_index(self.vocab[UNK])
        unk_vec = torch.mean(vec.vectors, dim=0).unsqueeze(0)
        pad_vec = torch.zeros(vec.vectors.shape[1]).unsqueeze(0)
        self.vectors = torch.cat([vec.vectors, unk_vec, pad_vec])
        if self.name:
            torch.save(self.vocab, self.data_path + '{}_vocab'.format(self.name))
            torch.save(self.vectors, self.data_path + self.name)
        else:
            torch.save(self.vocab, self.data_path + 'vector_vocab')
            torch.save(self.vectors, self.data_path + 'vectors')

    def load_vocab(self, seed=None):
        if self.name:
            self.vocab = torch.load(self.data_path + '{}_vocab'.format(self.name))
            self.vectors = torch.load(self.data_path + self.name)
        else:
            self.vocab = torch.load(self.data_path + 'vector_vocab')
            self.vectors = torch.load(self.data_path + 'vectors')


if __name__ == "__main__":
    dataProcessor = DataProcessor('pubmed')
    dataProcessor.split_data()
    # tokenizer = VectorTokenizer(vector_path='./data/glove', data_path='pubmed')
    # tokenizer.build_vocab()
    # print(tokenizer.vocab.get_stoi())
    # dataProcessor.get_tokenizer('glove', './data/glove')
    # dataProcessor.get_tokenizer()
    # dataloader = dataProcessor.get_dataloader()[2]
    # for idx, (content, value, lens, mask, id) in enumerate(dataloader):
    #     # if idx == 0:
    #     print(content)
    #     print(content[0].shape)
    #     print(value)
    #
    #     print(lens)
    dataProcessor.get_feature_graph('./data/glove')
