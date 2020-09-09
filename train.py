import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import collections

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import datetime

import resnet

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training in {device}')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

all_acc_dict = collections.OrderedDict()

def validate(model, train_loader, val_loader, epoch):
    accdict = []
    f = open("val.txt", "a")
    f.write(str(epoch) + '\n')
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        f.write("Accuracy {}: {:.2f}\n".format(name , correct / total))
    f.close()
    return accdict

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
            torch.save(net.state_dict(), 'models/firstmodel_' + str(epoch) + '_' +
                       str(loss_train / len(train_loader))+'.pt')
            validate(model, train_loader, val_loader, epoch)
net = resnet.ResNet50()
net = net.to(device)

# model = NetResDeep(n_chans1=32, n_blocks=100).to(device=device)
optimizer = optim.SGD(net.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs=1000,
    optimizer=optimizer,
    model=net,
    loss_fn=loss_fn,
    train_loader=train_loader,
)
