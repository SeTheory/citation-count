# 词嵌入和线性层构成的简易模型
import argparse
import datetime
import time

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from models.base_model import BaseModel
from torch import nn
import torch.nn.functional as F


class RNN(BaseModel):

    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, dropout=0.5, pad_size=1500,
                 rnn_model='LSTM', hidden_size=256, num_layers=2, bidirectional=False, **kwargs):
        super(RNN, self).__init__(vocab_size, embed_dim, num_class, pad_index, pad_size, word2vec, dropout)
        self.model_name = 'RNN_' + rnn_model
        # RNN模型参数设置
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # RNN模型初始化
        # self.encoder = self.embedding
        print('bidirectional', bidirectional)
        self.bidirectional = bidirectional
        self.encoder = RNNEncoder(rnn_model, 1, self.hidden_size, self.num_layers, dropout, self.bidirectional)
        self.decoder = RNNDecoder(rnn_model, 1, self.hidden_size, self.num_layers, dropout)

        self.drop_en = nn.Dropout(dropout)
        self.rnn_model = rnn_model
        # self.bn2 = nn.BatchNorm1d(hidden_size*2)
        # if self.bidirectional:
        #     self.fc = nn.Linear(hidden_size * 2, self.num_class)
        # else:
        #     self.fc = nn.Linear(hidden_size, self.num_class)
        # self.to(self.device)

    def forward(self, x, lengths, masks, ids, graph, **kwargs):
        # print(content.shape)
        content, inputs, valid_len = x
        x_embed = self.embedding(content)  # 不等长段落进行张量转化
        # x_embed = self.drop_en(x_embed)
        # 压缩向量
        packed_input = pack_padded_sequence(inputs[:, 0, :].unsqueeze(dim=-1).float(), valid_len.cpu().numpy(), batch_first=True, enforce_sorted=False)
        # 这里是输入的隐藏层也就是文本内容，这里直接依据序列长度做avgpool，然后和LSTM层数对齐
        h_0 = (x_embed.sum(dim=1)/lengths.unsqueeze(dim=-1)).unsqueeze(dim=0).repeat(self.num_layers * (int(self.bidirectional) + 1), 1, 1)
        if self.rnn_model == 'LSTM':
            c_0 = (x_embed.sum(dim=1)/lengths.unsqueeze(dim=-1)).unsqueeze(dim=0).repeat(self.num_layers * (int(self.bidirectional) + 1), 1, 1)
            hs = (h_0, c_0)
        else:
            hs = h_0

        return packed_input, hs


class RNNEncoder(nn.Module):
    def __init__(self, rnn_model, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(RNNEncoder, self).__init__()
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True,
                               bidirectional=bidirectional, num_layers=num_layers)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True,
                              bidirectional=bidirectional, num_layers=num_layers)
        else:
            print('No such RNN model!')

    def forward(self, packed_input, initial_state):
        # h_0, c_0 = initial_state
        packed_output, ht = self.rnn(packed_input, initial_state)
        out_rnn, lens = pad_packed_sequence(packed_output, batch_first=True)
        # lens = lens.unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, out_rnn.shape[-1])
        # output = torch.gather(out_rnn, 1, lens - 1) # 取最后一个hs的，暂时不用gather
        output = out_rnn
        return output, ht


class RNNDecoder(nn.Module):
    def __init__(self, rnn_model, input_size, hidden_size, num_layers, dropout):
        super(RNNDecoder, self).__init__()
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               dropout=dropout, batch_first=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              dropout=dropout, batch_first=True)
        else:
            print('No such RNN model!')
        self.fc = nn.Linear(hidden_size, 1)  # 因为是回归直接就是1了
        self.activation = nn.Tanh()

    def forward(self, cur_input, initial_state):
        # h_0, c_0 = initial_state
        out_rnn, ht = self.rnn(cur_input, initial_state)
        output = self.fc(self.activation(out_rnn))
        return output, ht


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    model = RNN(100, 10, 0, 0, hidden_size=10, num_layers=2)
    content = torch.tensor([list(range(10)), list(range(10, 20))])
    input_seq = torch.tensor([[[1, 2, 3, 0, 0],[0,0,0,0,0]],[[0, 1, 2, 3, 4], [0,0,0,0,0]]])
    valid_len = torch.tensor([3, 5])
    output_seq = torch.tensor([[4, 6, 8, 8, 8], [8, 12, 20, 32, 40]])
    masks = []
    lens = torch.tensor([5, 8])
    x = [content, input_seq, valid_len]
    ids = []
    graph = None
    pack, (h_0, c_0) = model(x, lens, masks, ids, graph)
    encoder_output, encoder_hidden = model.encoder(pack, (h_0, c_0))
    print(encoder_output.shape)
    print(encoder_output[0])
    print(encoder_hidden[0].shape)
    print(encoder_hidden[1].shape)

    decoder_input = torch.tensor([3, 4]).unsqueeze(dim=-1).unsqueeze(dim=-1).float()
    decoder_hidden = encoder_hidden
    decoder_output, (ht, ct) = model.decoder(decoder_input, decoder_hidden)
    print(decoder_output)
    print(ht.shape)

    decoder_params = list(map(id, model.decoder.parameters()))
    encoder_params = list(filter(lambda p: id(p) not in decoder_params, model.parameters()))
    print(len(decoder_params))
    print(len(encoder_params))
    print(len(list(model.parameters())))

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done simple_model!')
