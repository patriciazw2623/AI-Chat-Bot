import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (p, tag) in xy:
    bag = bag_of_words(p, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples

dataset = ChatDataSet()
bs = 8
train_loader = DataLoader(dataset = dataset, batch_size = bs, shuffle = True)

hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
learningrate = 0.001
num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr = learningrate)
for epoch in range(num_epochs):
    for (word, label) in train_loader:
        word = word.to(device)
        label = label.to(device)

        outputs = model(word)
        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')
print(f'final loss, loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict,
    "optimizer_state": optimizer.state_dict,
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
file = "data.pth"
torch.save(data, file)
torch.save(model.state_dict(), 'model_state.pth')


print(f"training complete. Model saved to {file}")