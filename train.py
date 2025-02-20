import gzip
import os
import pickle
import random

from matplotlib import pyplot as plt
from torchvision import transforms

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import MulticlassAccuracy
from model import Net

from inputs import INPUTS

BATCH_SIZE = 32  # mb size
EPOCHS = 10  # number of epochs
TRAIN_VAL_SPLIT = 0.85  # train/val ratio

DATA_DIR = 'data'
DATA_FILE = 'data.gzip'
MODEL_FILE = 'model.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# transformations for training/testing
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Grayscale(3),
    transforms.Resize((128,128)),
    #transforms.Pad((12, 12, 12, 0)),
    #transforms.CenterCrop(90),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
])

class TensorDatasetTransforms(torch.utils.data.TensorDataset):

    def __init__(self, x, y):
        super().__init__(x, y)

    def __getitem__(self, index):
        tensor = data_transform(self.tensors[0][index])
        return (tensor,) + tuple(t[index] for t in self.tensors[1:])

def read_data():
    """Read the data generated by keyboard_agent.py"""
    with gzip.open(os.path.join(DATA_DIR, DATA_FILE), 'rb') as f:
        data = pickle.load(f)

    #random.shuffle(data)

    # Fix for error when the env is init, it returns a tuple with observation it
    d = list()
    for i in range(len(data)):
        lst = list(data[i])
        #print( lst[0].shape())
        # if (type(lst[0]) is tuple):
        #     lst[0] = lst[0][0]
        if lst[1].sum() > 0: # Only accept where action count is > 0
            d.append(lst)  
    data = d

    states, actions, _, _, _ = map(np.array, zip(*data))
    act_classes = actions
    # act_classes = np.full((len(actions)), -1, dtype=int)
    # for i, a in enumerate(actions_set):
    #     act_classes[np.all(actions == a, axis=1)] = i

    # # drop unsupported actions
    # states = np.array(states)
    # states = states[act_classes != -1]
    # act_classes = act_classes[act_classes != -1]

    # for i, a in enumerate(actions_set):
    #     print("Actions of type {}: {}"
    #           .format(str(a), str(act_classes[act_classes == i].size)))

    print("Total transitions: ", len(states) ,str(len(act_classes)), act_classes[0], act_classes[1], act_classes[2])
    
    return states, act_classes


def create_datasets():

    x, y = read_data()

    x = np.moveaxis(x, 3, 1)  # channel first (torch requirement)

    # train dataset
    x_train = x[:int(len(x) * TRAIN_VAL_SPLIT)]
    y_train = y[:int(len(y) * TRAIN_VAL_SPLIT)]
    
    train_set = TensorDatasetTransforms(
        torch.tensor(x_train),
        torch.tensor(y_train))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=2)

    # test dataset
    x_val, y_val = x[int(len(x_train)):], y[int(len(y_train)):]

    val_set = TensorDatasetTransforms(
        torch.tensor(x_val),
        torch.tensor(y_val))

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             num_workers=2)

    return train_loader, val_loader

def create_ex_datasets():

    class TensorDatasetTransforms(torch.utils.data.TensorDataset):

        def __init__(self, x, y):
            super().__init__(x, y)

        def __getitem__(self, index):
            tensor = data_transform(self.tensors[0][index])
            r = (tensor,) + tuple(t[index] for t in self.tensors[1:])
            print(r)
            return r

    x, y = read_data()

    x = np.moveaxis(x, 3, 1)
    x_ex = x#[:2]
    y_ex = y#[:2]

    ex_set = TensorDatasetTransforms(
        torch.tensor(x_ex),
        torch.tensor(y_ex))

    ex_loader = torch.utils.data.DataLoader(ex_set, shuffle=True)

    return ex_loader


def train(model:nn.Module):
    """
    Training main method
    :param model: the network
    """
    #loss_function = nn.MSELoss() #
    #loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.MultiLabelMarginLoss()
    loss_function = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adagrad(model.parameters(), lr= 0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader, val_order = create_datasets()  # read datasets

    # train
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_epoch(model, loss_function, optimizer, train_loader)
        test(model, loss_function, val_order)

        # save model
        model_path = os.path.join(DATA_DIR, MODEL_FILE)
        torch.save(model.state_dict(), model_path)


def train_epoch(model, loss_function, optimizer, data_loader):
    """Train for a single epoch"""
    # set model to training mode
    model.train()

    current_loss = 0.0
    current_acc = 0
    metric = MulticlassAccuracy(num_classes=2)

    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):

        inputs:torch.Tensor = inputs

        # Show image
        # for data in np.moveaxis(np.array(inputs), 1, 3):
        #     plt.imshow(data, cmap='gray')
        #     plt.show(block=False)
        #     plt.pause(0.001)
        #     plt.draw()

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs.to(device).double()).to('cpu')
            loss = loss_function(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
        
        outputs = torch.sigmoid(outputs)
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        # statistics
        # outputs[outputs >= 0.9] = 1
        # outputs[outputs < 0.9] = 0
        # for o,l in zip(outputs, labels):
        #     metric.update(o, l)
        #     current_acc += metric.compute()#torch.sum(outputs == labels.data)
        # print(outputs * labels)
        # print(torch.sum(labels))
        current_acc += torch.sum(outputs * labels) / (torch.sum(labels) + torch.sum(outputs)) * 2
        current_loss += loss.item() * inputs.size(0)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))


def test(model, loss_function, data_loader):
    """Test over the whole dataset"""
    metric = MulticlassAccuracy(num_classes=2)
    model.eval()  # set model in evaluation mode

    current_loss = 0.0
    current_acc = 0

    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs.to(device).double()).to('cpu')
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs, labels)

        outputs = torch.sigmoid(outputs)
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        # for o,l in zip(outputs, labels):
        #     metric.update(o, l)
        #     current_acc += metric.compute()#torch.sum(outputs == labels.data)
        current_acc +=  torch.sum(outputs * labels) / torch.sum(labels)
        current_loss += loss.item() * inputs.size(0)

    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)

    print('Test Loss: {:.4f}; Accuracy: {:.4f}'
          .format(total_loss, total_acc))


if __name__ == '__main__':
    print('Training...')
    m = Net()
    m.to(device)
    m.eval()
    train(m)
    print('Training Done!')
    x_ex = create_ex_datasets()

    print('Outputs of Neural Network are as follows:')

    for i, (input, label) in enumerate(x_ex):

        for data in input:
            
            # data = np.moveaxis(data, 2, 0)  # channel first image
            # #_state = state
            # # numpy to tensor
            # data = torch.from_numpy(np.flip(data, axis=0).copy())
            d = data#data_transform(data)  # apply transformations
            d = d.unsqueeze(0)  # add additional dimension
            d = d.double()
        
            print("Example:",i+1)
            output = m(d.to(device).double()).to('cpu')
            output = torch.sigmoid(output)
            #output = torch.nn.functional.softmax(output)
            output = torch.round(output, decimals=2)
            print(output)
            print(label)

            d = np.moveaxis(np.array(data), 0, 2)
            plt.imshow(d)#, cmap='gray')
            plt.draw()
            plt.show()