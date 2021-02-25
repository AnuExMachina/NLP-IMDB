import pandas as pd 
import string
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

test['review'] = test['review'].str.lower()
train['review'] = train['review'].str.lower()

test['review'] = test['review'].str.replace('.', ' ')
train['review'] = train['review'].str.replace('.', ' ')

test['review'] = test['review'].str.replace(',', ' ')
train['review'] = train['review'].str.replace(',', ' ')

test['review'] = test['review'].str.replace('<br />', ' ')
train['review'] = train['review'].str.replace('<br />', ' ')

train['review'] = train['review'].str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
test['review'] = test['review'].str.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

X_train = train['review'].values.astype('U')
word_set = ' '.join(X_train)
word_set = word_set.split()
word_set = word_set + ['<sos>', '<eos>', '<unk>', '<pad>']
word_set = set(word_set)
dict = {}

for i, j in enumerate(word_set):
    dict[j] = i

y_train = train['target'].values
X_test = test['review'].values.astype('U')
y_test = test['target'].values
max_len = 20

X_train= np.array([(['<sos>'] + (i.split()[:max_len - 2] + ['<eos>'] + (max_len*['<pad>'])))[:max_len] for i in X_train])

for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        if X_train[i, j] not in dict.keys():
            X_train[i, j] = '<unk>'
        X_train[i, j] = dict[X_train[i, j]]

X_train = X_train.astype('int64')
X_test= np.array([(['<sos>'] + (i.split()[:max_len - 2] + ['<eos>'] + (max_len*['<pad>'])))[:max_len] for i in X_test])

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if X_test[i, j] not in dict.keys():
            X_test[i, j] = '<unk>'
        X_test[i, j] = dict[X_test[i, j]]

X_test = X_test.astype('int64')
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train.astype('float32'))
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test.astype('float32'))

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=True)

class NeuralNetwork(pl.core.LightningModule):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=75398, embedding_dim=100)
        self.lstm1 = nn.LSTMCell(input_size=100, hidden_size=64)
        self.lstm2 = nn.LSTMCell(input_size=64, hidden_size=64)
        self.dense1 = nn.Linear(in_features=64,out_features=128)
        self.dense2 = nn.Linear(in_features=128,out_features=64)
        self.dense3 = nn.Linear(in_features=64,out_features=1)

    def forward(self, inp):
        batch_size = inp.size(0)
        inp = self.embedding(inp)
        h1 = torch.zeros(batch_size, 64)
        c1 = torch.zeros(batch_size, 64)
        h2 = torch.zeros(batch_size, 64)
        c2 = torch.zeros(batch_size, 64)
        for i in range(inp.size(1)):
            x = inp[:, i, :]
            h1, c1 = self.lstm1(x, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
        x = F.gelu(self.dense1(h2))
        x = F.gelu(self.dense2(x))
        x = F.sigmoid(self.dense3(x))
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred.view(-1), y)
        self.log('train_accuracy',pl.metrics.functional.accuracy(torch.round(y_pred), y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred.view(-1), y)
        self.log('val_accuracy', pl.metrics.functional.accuracy(torch.round(y_pred), y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred.view(-1), y)
        self.log('test_accuracy',pl.metrics.functional.accuracy(torch.round(y_pred), y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.95)
        return [optimizer], [scheduler]


X, y = next(iter(dataset_train))
model = NeuralNetwork()
y_pred = model(X).view(-1)

trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, dataset_train, dataset_test)
trainer.test(model, dataset_test)

X, y = next(iter(dataset_test))
y_pred = torch.round(model(X).view(-1))