import dgl.graph_index

from models.RNN import *
from dgl.nn.pytorch.conv import APPNPConv


class RNNG(RNN):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, dropout=0.5, pad_size=1500,
                 rnn_model='LSTM', hidden_size=256, num_layers=2, bidirectional=False, **kwargs):
        super(RNNG, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, dropout, pad_size,
                                   rnn_model, hidden_size, num_layers, bidirectional, **kwargs)
        self.model_name = 'RNNG_' + rnn_model
        self.mix_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.gcn = APPNPConv(5, 0.2)
        self.activation = nn.Tanh()

    def forward(self, x, lengths, masks, ids, graph, **kwargs):
        # print(content.shape)
        content, inputs, valid_len = x
        x_embed = self.embedding(content)  # 不等长段落进行张量转化
        x_embed = x_embed.sum(dim=1) / lengths.unsqueeze(dim=-1)
        # x_embed = self.drop_en(x_embed)
        # 压缩向量
        # print(graph.ndata['h'])
        # 这里暂时只要静态图，考虑到预测的时候可能会有训练集里没有的节点，所以不修改graph_embed
        # 后续可以在图上应用各种方法，将原有的embed重新学习，只让新加的节点保持初始embed
        node_embed = graph.ndata['h'].detach()
        gnn_out = self.gcn(graph, node_embed)[ids]
        # print(gnn_out)
        mixed_embed = self.activation(self.mix_fc(torch.cat((x_embed, gnn_out), dim=-1)))
        packed_input = pack_padded_sequence(inputs[:, 0, :].unsqueeze(dim=-1).float(), valid_len.cpu().numpy(), batch_first=True, enforce_sorted=False)
        # 这里是输入的隐藏层也就是文本内容，这里直接依据序列长度做avgpool，然后和LSTM层数对齐
        h_0 = mixed_embed.unsqueeze(dim=0).repeat(self.num_layers * (int(self.bidirectional) + 1), 1, 1)
        if self.rnn_model == 'LSTM':
            c_0 = mixed_embed.unsqueeze(dim=0).repeat(self.num_layers * (int(self.bidirectional) + 1), 1, 1)
            hs = (h_0, c_0)
        else:
            hs = h_0

        return packed_input, hs



if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()
    model = RNNG(100, 10, 0, 0, hidden_size=10, num_layers=2)
    content = torch.tensor([list(range(10)), list(range(10, 20))])
    input_seq = torch.tensor([[[1, 2, 3, 0, 0],[0,0,0,0,0]],[[0, 1, 2, 3, 4], [0,0,0,0,0]]])
    valid_len = torch.tensor([3, 5])
    output_seq = torch.tensor([[4, 6, 8, 8, 8], [8, 12, 20, 32, 40]])
    masks = []
    lens = torch.tensor([5, 8])
    x = [content, input_seq, valid_len]
    graph = dgl.graph(([0,1,2,3,4],[1,1,1,1,1]), num_nodes=6)
    graph.ndata['h'] = torch.randn(6, 10)
    ids = torch.tensor([1, 0])

    pack, (h_0, c_0) = model(x, lens, masks, ids, graph)
    print(h_0.shape)
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

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done simple_model!')