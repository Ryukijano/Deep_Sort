#imports
import torch
import torchvision
import torch.nn as nn ## all neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Dropout, etc.
import torch.optim as optim ## all optimizers
import torch.nn.functional as F ## all functions
import torchvision.datasets as datasets ## all datasets
import torchvision.transforms as transforms ## all trasnforms 
from torch.utils.data import DataLoader ## to create DataLoader
from torch.utils.tensorboard import SummaryWriter ## to use tensorboard

#CNN

class CNN(nn.Module):
    def __init__(self, in_channels = 3, num_classes =10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
    
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
learning_rate = 0.001
in_channels = 3
num_classes = 10
batch_size = 512
num_epochs = 10

#load data
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#intialize network
model = CNN(in_channels=in_channels, num_classes=num_classes)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
writer  = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

for epoch in range(num_epochs):
    losses = []
    accuracies = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        #get data to cuda if available
        data, target = data.to(device), target.to(device)
        
        #forward pass
        scores = model(data)
        loss = criterion(scores, target)
        losses.append(loss.item())
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #calculate 'running' training accuracy
        _, predicted = torch.max(scores.data, 1)
        num_correct = (predicted == target).sum().item()
        running_train_acc = float(num_correct) / float(target.size(0))
        
        writer.add_scalar('Training Loss', loss, global_step = epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step = epoch * len(train_loader) + batch_idx)
    
    print (f'Epoch {epoch+1}/{num_epochs} Loss: {np.mean(losses):.4f} Accuracy: {np.mean(accuracies):.4f}')
        
        
        



        