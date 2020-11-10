import torch
from torch.utils.data import DataLoader
import random
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.autograd import Variable
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)
random.seed(1)
#데이터 로드
path = "D:/자동차 파손부위 분류"
batch_size = 10
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])
image_datasets = datasets.ImageFolder(path, transform)
train_size=int(0.9*len(image_datasets))
test_size = len(image_datasets)-train_size
class_names = image_datasets.classes
print(class_names)

train_dataset, test_dataset = torch.utils.data.random_split(image_datasets,[train_size,test_size])

train_loader= torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
test_loader= torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True, drop_last=True)


#모델
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)
model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
training_epochs=10
a=torch.floor_divide(len(train_loader.dataset),batch_size)

#학습
train_loss=[]
train_acc=[]
for epoch in range(training_epochs):
    trloss=0
    tracc=0
    for X, Y in train_loader:
        X = Variable(X).cuda()
        Y = Variable(Y).cuda()

        optimizer.zero_grad()
        hypothesis = model_ft(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        trloss+=cost
        predict=hypothesis.max(1)[1]
        acc=torch.true_divide((predict.eq(Y).sum())*100, batch_size)
        tracc += acc
    train_loss.append(torch.true_divide(trloss,a))
    train_acc.append(torch.true_divide(tracc,a))
    print('Epoch: {}] loss = {:.4f} acc:{:.4f}%'.format(epoch + 1, 
    torch.true_divide(trloss,a),torch.true_divide(tracc,a)))

# torch.save(model_ft, 'C:/Users/gombu/OneDrive/바탕 화면/3학년 2학기/캡스톤디자인1/세이브1.pt')

# model_ft = torch.load('C:/Users/gombu/OneDrive/바탕 화면/3학년 2학기/캡스톤디자인1/세이브1.pt')
# model_ft.eval()


correct =0
total = 0
with torch.no_grad():
    for data,target in test_loader:
        data=Variable(data).cuda()
        target=Variable(target).cuda()
        outputs=model_ft(data)
        _, predicted=torch.max(outputs.data,1)
        total+=target.size(0)
        correct+=(predicted==target).sum().item()
print('accuracy:%d %%'%(100*correct/total))



nb_classes = 6

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(test_loader):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)