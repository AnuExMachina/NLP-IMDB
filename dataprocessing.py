import os
import pandas as pd

test_neg_names = os.listdir('data/test/neg')
test_pos_names = os.listdir('data/test/pos')
train_neg_names = os.listdir('data/train/neg')
train_pos_names = os.listdir('data/train/pos')

test = []
for i in test_neg_names:
    with open(f'data/test/neg/{i}', 'r', encoding='utf-8') as f:
        data = f.read()
    test.append([data, 0])

for i in test_pos_names:
    with open(f'data/test/pos/{i}', 'r', encoding='utf-8') as f:
        data = f.read()
    test.append([data, 1])
test = pd.DataFrame(test, columns=['review', 'target']).sample(frac = 1)
test.to_csv('test.csv', index=False)

train = []
for i in train_neg_names:
    with open(f'data/train/neg/{i}', 'r', encoding='utf-8') as f:
        data = f.read()
    train.append([data, 0])

for i in train_pos_names:
    with open(f'data/train/pos/{i}', 'r', encoding='utf-8') as f:
        data = f.read()
    train.append([data, 1])

train = pd.DataFrame(train, columns=['review', 'target']).sample(frac = 1)
train.to_csv('train.csv', index=False)



