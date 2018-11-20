import torch 
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler

#from model import resnet
from PIL import Image
import os
import os.path
import six
import string
import sys
import time

#from DatasetLib.OQA import OQADatasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ", device)
print("Device number: ", torch.cuda.current_device())



transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.57420, 0.46984, 0.35669), (0.27862, 0.28578, 0.29748))
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.57420, 0.46984, 0.35669), (0.27862, 0.28578, 0.29748))
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.57420, 0.46984, 0.35669), (0.27862, 0.28578, 0.29748))
])

dataset_root = './dataset/'
trainset = datasets.ImageFolder(os.path.join(dataset_root, 'train'), transform_train)
#valset   = datasets.ImageFolder(os.path.join(dataset_root, 'validation'), transform_val)
testset  = datasets.ImageFolder(os.path.join(dataset_root, 'test'), transform_test)

dataset_sizes = {'train': len(trainset), 'test': len(testset)}

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
#valloader   = torch.utils.data.DataLoader(valset,   batch_size=64, shuffle=False,num_workers=4)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=1, shuffle=False,num_workers=4)

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)

model = model.to(device)

loss_list = []
epoch_list = []

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

model.train()
num_epochs = 75

since = time.time()
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    running_loss = 0.0
    running_corrects = 0
   
    for i, (inputs, labels) in enumerate(trainloader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects.double() / dataset_sizes['train']

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

    print()
    if epoch % 25 == 24:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #torch.save(model, "resnet18_"+str(epoch)+'.pt')

class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
class_table = list(list(0 for j in range(4)) for i in range(4))
classes = ('class1', 'class2', 'class3', 'class4')
food11_resnet18.eval()
tt = []
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    forward_time = time.time()
    outputs = food11_resnet18(inputs)
    tt.append(time.time()-forward_time)
    _, preds = torch.max(outputs, 1) #return value & indices
    c = (preds == labels).squeeze()

    for i in range(1):
        label = labels[i]
        pred  = preds[i]
        class_table[label][pred] += 1     
        class_correct[label] += c[i].item()
        class_total[label] += 1

for i in range(4):
    print(class_total[i])
    print('Accuracy of %7s : %2d %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]))

print('        class1 | class2 | class3 | class4')
for i in range(4):
    print('{:^8d}{:^6d} | {:^6d} | {:^6d} | {:^6d}'.format(i, class_table[i][0], class_table[i][1]
                                                      , class_table[i][2], class_table[i][3]))

print(np.mean(tt))

