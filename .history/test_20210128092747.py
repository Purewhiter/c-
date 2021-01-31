import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from logger import Logger

# 定义超参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 20


def to_np(x):
    return x.cpu().data.numpy()


# download datasets
train_dataset = datasets.CIFAR10(
    root='./cifar_data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.CIFAR10(
    root='./cifar_data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#define model
class slice_ssc(nn.Module):
    def __init__(self,in_channel,n_class):
        super(slice_ssc,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel,32,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(True),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(64*8*8,128),
            nn.Linear(128,64),
            nn.Linear(64,n_class))

    def forward(self,x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv2_out = conv2_out.view(conv2_out.size(0),-1)
        out = self.fc(conv2_out)
        return out

model = slice_ssc(1,10)
print (model)

use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
logger = Logger('./logs')
#training
for epoch in range(num_epoches):
    print ('epoch {}').format(epoch+1)
    train_loss=0.0
    train_acc=0.0

    #==========training============
    for i,data in enumerate(train_loader,1):
        img,label=data
        img=img.view(img.size(0)*3,1,32,32)
        label = torch.cat((label,label,label),0)
        #print img.size()
        #print label.size()
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)      

        #forward
        out = model(img)
        loss = criterion(out,label)
        train_loss += loss.data[0] #*label.size(0)
        _, pred = torch.max(out,1)
        train_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        train_acc += train_correct.data[0]
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #=============log===============
        step = epoch*len(train_loader)+i
        info = {'loss':loss.data[0],'accuracy':accuracy.data[0]}   
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)

        info = {'images': to_np(img.view(-1, 32, 32)[:10])}
        for tag, images in info.items():
            logger.image_summary(tag, images, step)
        if i % 300 == 0:
            print( '[{}/{}] Loss: {:.6f}, Acc: {:.6f}').format(epoch + 1, num_epoches, train_loss / (batch_size * i),train_acc / (batch_size * i))

    print ('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}').format(epoch + 1, train_loss / (len(train_dataset)), train_acc / (len(train_dataset)))

    #============testing=============
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for data in test_loader:
        img,label = data
        img=img.view(img.size(0)*3,1,32,32)
        label = torch.cat((label,label,label),0)
        if use_gpu:
            img = Variable(img,volatile=True).cuda()
            label = Variable(label,volatile=True).cuda()
        else:
            img = Variable(img, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data[0]
    print( 'Test Loss: {:.6f}, Acc: {:.6f}').format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset)))

# 保存模型
torch.save(model.state_dict(), './cnn.pth')