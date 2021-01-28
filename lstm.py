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
# len(word_set)
word_set = set(word_set)

dict = {}
for i, j in enumerate(word_set):
    dict[j] = i

y_train = train['target'].values

X_test = test['review'].values.astype('U')

y_test = test['target'].values

max_len = 40

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

# X_train.size()

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)


dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=True)


class NeuralNetwork(pl.core.LightningModule):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=75398, embedding_dim=100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=20, num_layers=2, batch_first=True, dropout=0.25)
        self.dropout1 = nn.Dropout(0.25)
        self.normalize = nn.LayerNorm(20)
        self.dense1 = nn.Linear(in_features=20,out_features=80)
        self.dropout2 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(in_features=80,out_features=64)
        self.dropout3 = nn.Dropout(0.25)
        self.dense3 = nn.Linear(in_features=64,out_features=1)
    def forward(self, inp):
        batch_size = inp.size(0)
        x = self.embedding(inp)
        h = torch.zeros(2, batch_size, 20)
        c = torch.zeros(2, batch_size, 20)
        x, (h, c) = self.lstm(x, (h, c))
        x = x[:, -1, :]
        x = self.dropout1(x)
        x = self.normalize(x)
        x = F.gelu(self.dense1(x))
        x = self.dropout2(x)
        x = F.gelu(self.dense2(x))
        x = self.dropout3(x)
        x = F.sigmoid(self.dense3(x))
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred.view(-1), y)
        self.log('train_accuracy',pl.metrics.functional.auroc(y_pred, y), prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred.view(-1), y)
        self.log('val_accuracy', pl.metrics.functional.auroc(y_pred, y), prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.binary_cross_entropy(y_pred.view(-1), y)
        self.log('test_accuracy',pl.metrics.functional.auroc(y_pred, y), prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.95)
        return [optimizer], [scheduler]
    



X, y = next(iter(dataset_train))
model = NeuralNetwork()
# y.shape
y_pred = model(X)
# type(y_pred)
# y_pred[:, -1, :].view(-1).shape
# y_pred[0].shape==y_pred[1].shape
#y.shape()
#X.shape

#y_pred.shape


trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, dataset_train, dataset_test)
trainer.test(model, dataset_test)
# torch.nn.Embedding(num_embeddings: 75398, embedding_dim: 100)

X, y = next(iter(dataset_test))
y_pred = torch.round(model(X).view(-1))
