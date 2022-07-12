from models.RNN import *
from dgl.nn.pytorch.conv import APPNPConv


class RNNG(RNN):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, word2vec=None, keep_prob=0.5, pad_size=1500,
                 rnn_model='LSTM', hidden_size=256, num_layers=2, bidirectional=False, **kwargs):
        super(RNNG, self).__init__(vocab_size, embed_dim, num_class, pad_index, word2vec, keep_prob, pad_size,
                                   rnn_model, hidden_size, num_layers, bidirectional, **kwargs)
        self.mix_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.gcn = None

    def forward(self, x, lengths, masks, ids, graph, **kwargs):
        # print(content.shape)
        content, inputs, valid_len = x
        x_embed = self.embedding(content)  # 不等长段落进行张量转化
        # x_embed = self.drop_en(x_embed)
        # 压缩向量
        packed_input = pack_padded_sequence(inputs.unsqueeze(dim=-1).float(), valid_len.cpu().numpy(), batch_first=True, enforce_sorted=False)
        # 这里是输入的隐藏层也就是文本内容，这里直接依据序列长度做avgpool，然后和LSTM层数对齐
        h_0 = x_embed.mean(dim=1).unsqueeze(dim=0).repeat(self.num_layers * (int(self.bidirectional) + 1), 1, 1)
        c_0 = x_embed.mean(dim=1).unsqueeze(dim=0).repeat(self.num_layers * (int(self.bidirectional) + 1), 1, 1)

        return packed_input, (h_0, c_0)
