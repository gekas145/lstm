import torch
import zipfile
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lstm import SimpleNet


nbatch = 8
epochs = 10


def load_mnist():
    def prepare_images(buffer):
        prepared = np.frombuffer(buffer, offset=16, dtype=np.uint8).astype(np.double)
        return prepared.reshape(len(buffer) // image_size ** 2, image_size * image_size) / 255

    image_size = 28

    file_names = ['t10k-images.idx3-ubyte',
                  't10k-labels.idx1-ubyte',
                  'train-images.idx3-ubyte',
                  'train-labels.idx1-ubyte']
    data = []
    with zipfile.ZipFile('data/MNIST.zip', 'r') as zf:
        for file in file_names:
            with zf.open(file, 'r') as f:
                data.append(f.read())

    X = prepare_images(data[0])
    y = np.frombuffer(data[1], offset=8, dtype=np.uint8).astype(np.int64)

    idxs = np.random.choice(range(len(y)), len(y), replace=False)
    X_test, X_train = X[idxs[0:3000]], X[idxs[3000:]]
    y_test, y_train = y[idxs[0:3000]], y[idxs[3000:]]

    return torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test), torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)

class MNISTDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_target(y_batch):
    target = torch.zeros((len(y_batch), 10))
    target[range(len(y_batch)), y_batch] = 1.0
    return target

def get_loss(X, y):
    outputs = net(X.view(nbatch, 28, 28))[:, -1, :]
    target = get_target(y)

    return criterion(outputs, target)

def get_accuracy(X, y):
    outputs = net(X.view(nbatch, 28, 28))[:, -1, :]
    pred = torch.argmax(outputs, axis=-1)

    return torch.sum(pred == y)

X_test, y_test, X_train, y_train = load_mnist()
training = DataLoader(MNISTDataset(X_train, y_train), batch_size=nbatch, shuffle=True)
validation = DataLoader(MNISTDataset(X_test, y_test), batch_size=nbatch)

net = SimpleNet(28, 512, 10, proj_dim=128)

criterion = torch.nn.CrossEntropyLoss(reduction="sum")

optimizer = torch.optim.Adam(net.parameters())

start_time = time.time()

for epoch in range(epochs):
    
    for X_batch, y_batch in training:
        optimizer.zero_grad()

        loss = get_loss(X_batch, y_batch)
        loss /= X_batch.shape[0]
        loss.backward()

        optimizer.step()


    with torch.inference_mode():
        train_acc, valid_acc = 0.0, 0.0
        for train_dt, valid_dt in zip(training, validation):
            train_acc += get_accuracy(*train_dt)
            valid_acc += get_accuracy(*valid_dt)

        print(f"[Epoch: {epoch+1}] train acc: {100 * train_acc.data/len(validation.dataset):.2f}, validation acc: {100 * valid_acc.data/len(validation.dataset):.2f}")


print(f"Learning time: {time.time() - start_time:.3f} seconds")

with torch.no_grad():
    X_batch, y_batch = X_test[0:10], y_test[0:10]

    pred = net(X_batch.view(10, 28, 28))[:, -1, :]
    pred = torch.softmax(pred, axis=-1).numpy()

    for x, y in zip(pred, y_batch):
        print("Pred:", x, "\nPred class:", np.argmax(x), "Real:", y)
        print("================")