import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from data.data_loader import cifar10, cifar100
from lib.util import TransformTwice
from models import *

# Load CIFAR100 dataset with enhanced data augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
])
train_transform = TransformTwice(train_transform, train_transform)

trainset = cifar100(root='/data/temp_zenglin/data',
                    train=True,
                    classes=range(100),
                    download=True,
                    transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))
])
test_transform = TransformTwice(test_transform, test_transform)

testset = cifar100(root='/data/temp_zenglin/data',
                   train=False,
                   classes=range(100),
                   download=True,
                   transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Initialize and load the model
model = resnet32_cifar(num_classes=100).cuda()
# model.apply(lambda m: nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)

# Define loss function and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the network with early stopping
best_accuracy = 0
for epoch in range(10):  # Increase number of epochs
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[-1].to(device)

        optimizer.zero_grad()   # zero the parameter gradients

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()         # compute gradients
        optimizer.step()        # update parameters

        running_loss += loss.item()
    scheduler.step()
    print('Epoch %d: Loss is %.3f' % (epoch + 1, running_loss / len(trainloader)))  # print statistics

    # Evaluate the network
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[-1].to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on test images: %d %%' % accuracy)   # print statistics
    
    # Save the best model
    # if accuracy > best_accuracy:
    #     best_accuracy = accuracy
    #     torch.save(model.state_dict(), 'models/resnet32_cifar100_best.pth')
torch.save(model.state_dict(), 'models/resnet32_cifar100_epoch10.pth')
print('Finished Training')