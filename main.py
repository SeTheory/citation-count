from data_processor import *
from models.RNN import RNN

dataProcessor = DataProcessor('pubmed')
dataProcessor.get_tokenizer()
dataProcessor.get_dataloader(4)

model = RNN(len(dataProcessor.tokenizer.vocab), 128, 0, dataProcessor.tokenizer.vocab[PAD], hidden_size=128, num_layers=2)
# model = model.to(model.device)
model.train_batch(dataProcessor.dataloaders, 50, lr=10e-3, record_path='./results/pubmed/')