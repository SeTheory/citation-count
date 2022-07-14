import argparse
import datetime

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from models.RNN import RNN


class RNNN(RNN):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, dropout=0.5, pad_size=1500,
                 rnn_model='LSTM', hidden_size=256, num_layers=2, bidirectional=False, **kwargs):
        super(RNNN, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, dropout, pad_size,
                                   rnn_model, hidden_size, num_layers, bidirectional, **kwargs)

    def forward(self, x, lengths, masks, ids, graph, **kwargs):
        # print(content.shape)
        content, inputs, valid_len = x
        # x_embed = self.embedding(content)  # 不等长段落进行张量转化
        # x_embed = self.drop_en(x_embed)
        # 压缩向量
        packed_input = pack_padded_sequence(inputs[:, 0, :].unsqueeze(dim=-1).float(), valid_len.cpu().numpy(), batch_first=True,
                                            enforce_sorted=False)

        return packed_input, None

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    model = RNN(100, 10, 0, 0, hidden_size=10, num_layers=2)
    content = torch.tensor([list(range(10)), list(range(10, 20))])
    input_seq = torch.tensor([[1, 2, 3, 0, 0], [0, 1, 2, 3, 4]])
    valid_len = torch.tensor([3, 5])
    output_seq = torch.tensor([[4, 6, 8, 8, 8], [8, 12, 20, 32, 40]])
    masks = []
    lens = torch.tensor([5, 8])
    x = [content, input_seq, valid_len]
    pack, hs = model(x, lens, masks)
    encoder_output, encoder_hidden = model.encoder(pack, hs)
    print(encoder_output.shape)
    print(encoder_output[0])
    print(encoder_hidden[0].shape)
    print(encoder_hidden[1].shape)

    decoder_input = torch.tensor([3, 4]).unsqueeze(dim=-1).unsqueeze(dim=-1).float()
    decoder_hidden = encoder_hidden
    decoder_output, hs = model.decoder(decoder_input, decoder_hidden)
    print(decoder_output)
    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done simple_model!')

