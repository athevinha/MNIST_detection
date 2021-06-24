import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
# Hyper-parameters
input_size = 3136    # 784
num_classes = 10
num_epochs = 1
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size)

        # Forward pass
        y_hat = model(images)

        loss = criterion(y_hat, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Xac suat:{} %'.format(
        100 * correct / total))
img_dir = './number.jpg'
img = plt.imread(img_dir)
img = torch.from_numpy(img)
# crop images
img = img.convert('RGB')
cropped_image1 = img[:, 16:40]
cropped_image2 = img[:, 40:60]
cropped_image3 = img[:, 60:80]
cropped_image4 = img[:, 80:110]
cropped_image5 = img[:, 16:40]
print(cropped_image1)
# plt.imshow(cropped_image1)
# # plt.imshow(cropped_image1)
# # plt.imshow(cropped_image2)
# # plt.imshow(cropped_image3)
# #
# plt.imshow(cropped_image1)
# images = cropped_image1.reshape(-1, input_size)
# outputs = model(images)
# _, predicted = torch.max(outputs.data, 1)

# print("Số được nhận biết: ", predicted[i].item())
# plt.show()
