import argparse
import datetime
import math
import random
import logging

import numpy as np
import pandas as pd

from torch import nn
import time
import torch
import torch.nn.functional as F

from utilis.scripts import eval_result, result_format


class BaseModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, pad_index, pad_size=1500, word2vec=None, keep_prob=0.5,
                 model_path=None, **kwargs):
        super(BaseModel, self).__init__()
        # 最基本的模型参数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not model_path:
            if word2vec is not None:
                self.embedding = nn.Embedding.from_pretrained(word2vec, freeze=False)
            else:
                self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False, padding_idx=pad_index)
        self.embed_dim = embed_dim
        self.num_class = num_class
        self.pad_size = pad_size
        self.model_name = 'base_model'
        self.adaptive_lr = False
        self.warmup = False
        self.T = kwargs['T'] if 'T' in kwargs.keys() else 400
        self.seed = None
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
        if self.warmup:
            print('warm up T', self.T)
        # 模型学习相关设置（写在训练方法可能更好）
        # self.epochs = 10
        # self.lr = 5
        # self.batch_size = 64  # 实际上在数据处理中就设定完毕

    def init_weights(self):
        # 初始化词嵌入权重
        print('init')

    def forward(self, text, lengths, masks, **kwargs):
        # 运行前向函数
        return 'forward'

    def train_model(self, dataloader, epoch, criterion, encoder_optimizer, decoder_optimizer):
        """
        最基本的模型训练方式，包括数据读取、轮次、损失函数和优化器
        根据需求进行扩展，添加相应的参数
        """
        self.train()
        total_acc, total_count = 0, 0
        log_interval = 10
        start_time = time.time()
        loss_list = []
        all_predicted_values = []
        all_true_values = []
        teacher_forcing_ratio = 0.5
        all_loss = 0

        for idx, (x, values, lengths, masks, ids) in enumerate(dataloader):
            loss = 0
            # 转移数据
            if type(x) == list:
                x = [content.to(self.device) for content in x]
            else:
                x = x.to(self.device)
            # print('io-time:', time.time()-start_time)
            values = values.to(self.device)
            lengths = lengths.to(self.device)
            masks = masks.to(self.device)
            # print('io-time:', time.time() - start_time)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            packed_input, (h_0, c_0) = self(x, lengths, masks)
            encoder_output, encoder_hidden = self.encoder(packed_input, (h_0, c_0))

            decoder_input = values[:, 0, -1].unsqueeze(dim=-1).unsqueeze(dim=-1).float()
            decoder_hidden = encoder_hidden
            target_tensor = values[:, 1, :].unsqueeze(dim=-1).float()
            target_length = target_tensor.shape[1]
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, target_tensor[:, di].unsqueeze(dim=1))
                    all_predicted_values.append(decoder_output.squeeze().detach().cpu())
                    all_true_values.append(target_tensor[:, di].squeeze().detach().cpu())
                    decoder_input = target_tensor[:, di].unsqueeze(dim=1)  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                    # topv, topi = decoder_output.topk(1)
                    # decoder_input = topi.squeeze().detach()  # detach from history as input
                    decoder_input = decoder_output

                    loss += criterion(decoder_output, target_tensor[:, di].unsqueeze(dim=1))
                    all_predicted_values.append(decoder_output.squeeze().detach().cpu())
                    all_true_values.append(target_tensor[:, di].squeeze().detach().cpu())

            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)  # 梯度裁剪，防止梯度消失或爆炸
            # if self.warmup:
            #     self.scheduler.step()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print_loss = loss.item() / target_length
            all_loss += print_loss
            rmse = math.sqrt(print_loss)

            if idx % log_interval == 0 and idx > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| RMSE {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                                 rmse, print_loss))
                # epoch_log.write('| epoch {:3d} | {:5d}/{:5d} batches '
                #                 '| accuracy {:8.3f} | loss {:8.3f}\n'.format(epoch, idx, len(dataloader),
                #                                                              total_acc / total_count, loss.item()))
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| RMSE {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(dataloader),
                                                                 rmse, print_loss))
                total_acc, total_count = 0, 0
        elapsed = time.time() - start_time
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, elapsed))
        logging.info('-' * 59)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, elapsed))
        all_predicted_values = torch.stack(all_predicted_values).numpy().T
        all_true_values = torch.stack(all_true_values).numpy().T
        avg_loss = all_loss / len(dataloader)
        results = [avg_loss] + eval_result(all_true_values, all_predicted_values)
        format_str = result_format(results)
        for line in format_str.split('\n'):
            logging.info(line)

        return results

    # def build(self, vocab_size, embed_dim, num_class):
    #     self.model_name = 'base_model'

    def train_batch(self, dataloaders, epochs, lr=10e-4, criterion='MSE', optimizer='ADAM',
                    scheduler=False, record_path=None, save_path=None):
        """
        整体多次训练模型，先更新参数再用验证集测试
        输入主要包括训练集和验证集，训练轮次，损失函数，优化方法，自动调整学习率方法
        """
        final_results = []
        train_dataloader, val_dataloader, test_dataloader = dataloaders
        total_accu = None
        val_accu_list = []
        # 损失函数设定
        if criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()
        elif criterion == 'MSE':
            criterion = nn.MSELoss()
        self.criterion = criterion
        # 优化器设定
        if optimizer == 'SGD':
            encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=lr, weight_decay=1e-3)
            decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=lr, weight_decay=1e-3)
        elif optimizer == 'ADAM':
            if self.adaptive_lr:
                encoder_optimizer = torch.optim.Adam(self.encoder.get_group_parameters(lr), lr=lr, weight_decay=0)
                decoder_optimizer = torch.optim.Adam(self.decoder.get_group_parameters(lr), lr=lr, weight_decay=0)
            else:
                encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=1e-3)
                decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=1e-3)
        elif optimizer == 'ADAMW':
            if self.adaptive_lr:
                encoder_optimizer = torch.optim.AdamW(self.encoder.get_group_parameters(lr), lr=lr)
                decoder_optimizer = torch.optim.AdamW(self.decoder.get_group_parameters(lr), lr=lr)
            else:
                encoder_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr)
                decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        # 学习时自动调整学习速度
        # if self.warmup:
        #     self.scheduler = TriangularScheduler(optimizer, cut_frac=0.1, T=self.T, ratio=32)
        #     optimizer.zero_grad()
        #     optimizer.step()

        if self.seed:
            print('{}_records_{}.csv'.format(self.model_name, self.seed))
            fw = open(record_path + '{}_records_{}.csv'.format(self.model_name, self.seed), 'w')
        else:
            fw = open(record_path + '{}_records.csv'.format(self.model_name), 'w')

        fw.write('epoch' + ',{}_loss,{}_mae,{}_r2,{}_mse,{}_rmse'.replace('{}', 'train')
                 + ',{}_loss,{}_mae,{}_r2,{}_mse,{}_rmse'.replace('{}', 'val')
                 + ',{}_loss,{}_mae,{}_r2,{}_mse,{}_rmse'.replace('{}', 'test') + '\n')

        for epoch in range(1, epochs + 1):
            # epoch_log = open(record_path + '{}_epoch_{}.log'.format(self.model_name, epoch), 'w')
            # print(record_path + '{}_epoch_{}.log'.format(self.model_name, epoch))
            logging.basicConfig(level=logging.INFO,
                                filename=record_path + '{}_epoch_{}.log'.format(self.model_name, epoch),
                                filemode='w+',
                                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                                force=True)
            epoch_log = None
            epoch_start_time = time.time()
            train_results = self.train_model(train_dataloader, epoch, criterion, encoder_optimizer, decoder_optimizer)
            val_results = self.evaluate(val_dataloader)
            test_results = self.test(test_dataloader)
            all_results = train_results + val_results + test_results
            fw.write(','.join([str(epoch)] + [str(round(x, 6)) for x in all_results]) + '\n')
            # acc, prec, recall, maf1, f1, auc, log_loss_value = val_results
            # val_accu_list.append(round(acc, 3))
            # if save_path:
            #     self.save_model(save_path + '{}_{}.pkl'.format(self.model_name, epoch))


        fw.close()
        return final_results

    def save_model(self, path):
        torch.save(self, path)
        print('Save successfully!')

    def load_model(self, path):
        model = torch.load(path)
        print('Load successfully!')
        return model

    def evaluate(self, dataloader, phase='val'):
        self.eval()
        all_true_values = []
        all_predicted_values = []
        all_loss = 0
        start_time = time.time()

        with torch.no_grad():
            for idx, (x, values, lengths, masks, ids) in enumerate(dataloader):
                loss = 0
                # 转移数据
                if type(x) == list:
                    x = [content.to(self.device) for content in x]
                else:
                    x = x.to(self.device)
                # print('io-time:', time.time()-start_time)
                values = values.to(self.device)
                lengths = lengths.to(self.device)
                masks = masks.to(self.device)

                packed_input, (h_0, c_0) = self(x, lengths, masks)
                encoder_output, encoder_hidden = self.encoder(packed_input, (h_0, c_0))

                decoder_input = values[:, 0, -1].unsqueeze(dim=-1).unsqueeze(dim=-1).float()
                decoder_hidden = encoder_hidden
                target_tensor = values[:, 1, :].unsqueeze(dim=-1).float()
                target_length = target_tensor.shape[1]

                for di in range(target_length):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden)
                    # topv, topi = decoder_output.topk(1)
                    # decoder_input = topi.squeeze().detach()  # detach from history as input
                    decoder_input = decoder_output

                    loss += self.criterion(decoder_output, target_tensor[:, di].unsqueeze(dim=1)).item() / target_length
                    all_loss += loss
                    all_predicted_values.append(decoder_output.squeeze().detach().cpu())
                    all_true_values.append(target_tensor[:, di].squeeze().detach().cpu())

            elapsed = time.time() - start_time
            print('-' * 59)
            print('| end of {} | time: {:5.2f}s |'.format(phase, elapsed))
            logging.info('-' * 59)
            logging.info('| end of {} | time: {:5.2f}s |'.format(phase, elapsed))
            all_predicted_values = torch.stack(all_predicted_values).numpy().T
            all_true_values = torch.stack(all_true_values).numpy().T
            avg_loss = all_loss / len(dataloader)
            results = [avg_loss] + eval_result(all_true_values, all_predicted_values)
            format_str = result_format(results)
            for line in format_str.split('\n'):
                logging.info(line)

            return results

    def test(self, test_dataloader, phase='test', epoch_log=None):
        results = self.evaluate(test_dataloader, phase)
        return results

    # def get_mistake_results(self, test_dataloader):
    #     self.test(test_dataloader)
    #     self.eval()
    #     mistake_results = []
    #
    #     with torch.no_grad():
    #         for idx, (contents, labels, lengths, masks, indexes) in enumerate(test_dataloader):
    #             if type(contents) == list:
    #                 contents = [content.to(self.device) for content in contents]
    #             else:
    #                 contents = contents.to(self.device)
    #             labels = labels.to(self.device)
    #             lengths = lengths.to(self.device)
    #             masks = masks.to(self.device)
    #
    #             model_result = self(contents, lengths, masks)
    #             if type(model_result) == list:
    #                 model_result = model_result[0]
    #             predicted_result = F.softmax(model_result, dim=1).detach().cpu().numpy()
    #             predicted_label = np.array(predicted_result).argmax(1)
    #             true_label = labels.cpu().numpy().tolist()
    #             result = zip(predicted_result, predicted_label, true_label, indexes.cpu().numpy().tolist(),
    #                          (predicted_label == true_label))
    #             mistake_results.extend(list(filter(lambda x: x[-1] == False, result)))
    #
    #     return mistake_results


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')

    args = parser.parse_args()

    if args.phase == 'test':
        print('This is a test process.')
    else:
        print('error! No such method!')
    end_time = datetime.datetime.now()
    print('{} takes {} seconds'.format(args.phase, (end_time - start_time).seconds))

    print('Done base_model!')
