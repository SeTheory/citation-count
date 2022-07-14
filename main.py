import argparse
import datetime
import os

import dgl
import numpy as np

from data_processor import *
from models.RNN import RNN
from models.RNNG import RNNG
from models.RNNN import RNNN

from utilis.scripts import get_configs


def get_model(model_name, config, vectors=None):
    model = None
    if model_name == 'LSTM':
        model = RNN(config['vocab_size'], config['hidden_size'], 0, config['pad_idx'],
                    word2vec=vectors, dropout=config['dropout'], rnn_model='LSTM',
                    hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    elif model_name == 'GRU':
        model = RNN(config['vocab_size'], config['hidden_size'], 0, config['pad_idx'],
                    word2vec=vectors, dropout=config['dropout'], rnn_model='GRU',
                    hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    elif model_name == 'LSTMG':
        model = RNNG(config['vocab_size'], config['hidden_size'], 0, config['pad_idx'],
                     word2vec=vectors, dropout=config['dropout'], rnn_model='LSTM',
                     hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    elif model_name == 'GRUG':
        model = RNNG(config['vocab_size'], config['hidden_size'], 0, config['pad_idx'],
                     word2vec=vectors, dropout=config['dropout'], rnn_model='GRU',
                     hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    elif model_name == 'LSTMN':
        model = RNNN(config['vocab_size'], config['hidden_size'], 0, config['pad_idx'],
                    word2vec=vectors, dropout=config['dropout'], rnn_model='LSTM',
                    hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    elif model_name == 'GRUN':
        model = RNNN(config['vocab_size'], config['hidden_size'], 0, config['pad_idx'],
                    word2vec=vectors, dropout=config['dropout'], rnn_model='GRU',
                    hidden_size=config['hidden_size'], num_layers=config['num_layers'])
    model.to(model.device)
    return model


def train_single(data_source, model_name, config, seed=123, norm=False):
    dataProcessor = DataProcessor(data_source, max_len=config['max_len'], seed=seed, norm=norm)
    dataProcessor.get_tokenizer(config['tokenizer_type'], config['tokenizer_path'])
    dataProcessor.get_dataloader(config['batch_size'])
    if config['use_graph']:
        graph_dict = dataProcessor.load_graph(config['graph_name'])
    else:
        graph_dict = None

    record_path = './results/{}/'.format(data_source)
    save_path = './checkpoints/{}/'.format(data_source)

    config['vocab_size'] = len(dataProcessor.tokenizer.vocab)
    config['pad_idx'] = dataProcessor.tokenizer.vocab[PAD]
    model = get_model(model_name, config, dataProcessor.tokenizer.vectors)
    model.train_batch(dataloaders=dataProcessor.dataloaders, epochs=config['epochs'],
                      lr=config['lr'], criterion=config['criterion'], optimizer=config['optimizer'],
                      record_path=record_path, save_path=save_path, graph=graph_dict)


def setup_seed(seed):
    # 设定随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dgl.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # os.environ["OMP_NUM_THREADS"] = '1'


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')

    parser.add_argument('--phase', default='LSTMN', help='the function name.')
    parser.add_argument('--ablation', default=None, help='the ablation modules.')
    parser.add_argument('--data_source', default='pubmed', help='the data source.')
    parser.add_argument('--norm', default=False, help='the data norm.')
    parser.add_argument('--mode', default=None, help='the model mode.')
    parser.add_argument('--type', default='BERT', help='the model type.')
    parser.add_argument('--seed', default=123, help='the data seed.')
    parser.add_argument('--model_seed', default=123, help='the model seed.')
    parser.add_argument('--model', default=None, help='the selected model for other methods.')
    parser.add_argument('--model_path', default=None, help='the selected model for deep analysis.')

    args = parser.parse_args()
    print('args', args)
    print('data_seed', args.seed)
    # setup_seed(int(args.seed))  # 原本的模型种子和数据种子一同修改
    MODEL_SEED = int(args.model_seed)
    setup_seed(MODEL_SEED)  # 模型种子固定为123，数据种子根据输入修改
    print('model_seed', MODEL_SEED)

    model_list = ['LSTM', 'GRU', 'LSTMG', 'GRUG', 'LSTMN', 'GRUN']
    configs = get_configs(args.data_source, model_list)

    if args.phase in model_list:
        train_single(args.data_source, args.phase, configs[args.phase], args.seed, args.norm)
        print('{} done'.format(args.phase))

