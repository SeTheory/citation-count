from data_processor import *
from models.RNN import RNN

dataProcessor = DataProcessor('pubmed')
dataProcessor.split_data()
dataProcessor.get_tokenizer()
dataProcessor.get_dataloader(32)

model = RNN(len(dataProcessor.tokenizer.vocab), 300, 0, dataProcessor.tokenizer.vocab[PAD], hidden_size=300, num_layers=2)
# model = model.to(model.device)
model.train_batch(dataProcessor.dataloaders, 50, lr=1e-3, record_path='./results/pubmed/')