import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
# xHeight = len(inputX)
# xWidth = len(inputX[0])
# # sobel
# gx = [[-1, 0, 1],
#       [-2, 0, 2],
#       [-1, 0, 1]]
# gy = [[1, 2, 1],
#       [0, 0, 0],
#       [-1, -2, -1]]
# yHeight = (len(inputX) - len(gx))
# yWidth = (len(inputX[0]) - len(gx[0]))
# OutputY1 = np.zeros((yHeight, yWidth))
# OutputY2 = np.zeros((yHeight, yWidth))

# for i in range(yHeight):
#     for j in range(yWidth):
#         OutputY1[i, j] = np.sum(inputX[i: i+3, j:j+3] * gx)
#         # nhan covution
# for i in range(yHeight):
#     for j in range(yWidth):
#         OutputY2[i, j] = np.sum(inputX[i: i+3, j:j+3] * gy)
#         # nhan covution
# # plt.imshow(np.sqrt(OutputY1**2 + OutputY2**2), cmap="gray")
# # plt.show()

# Hyper-parameters
input_size = 28 * 28    # 784
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# # MNIST dataset (images and labels)
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
print(enumerate(train_loader))
# Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Reshape images to (batch_size, input_size)
#         images = images.reshape(-1, input_size)
#         # Forward pass

#         y_hat = model(images)
#         print("Label :", labels[0])
#         print("y_hat :", y_hat)
#         loss = criterion(y_hat, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, input_size)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()

#     print('Xac suat:{} %'.format(
#         100 * correct / total))
# # read images
# model = torch.load('model.ckpt')
# inputX = cv2.imread('number.jpg')
# inputX = cv2.cvtColor(inputX, cv2.COLOR_BGR2GRAY)
# # print...
# print(inputX)
# # crop number
# img_crop = inputX[:, 60:90]
# # rezi number to 28 x 28
# img_resized = cv2.resize(src=img_crop, dsize=(28, 28))
# # convert numpy to tosensor
# img_resized = torch.from_numpy(img_resized)
# # show images
# plt.imshow(img_resized,cmap="gray")
# # resahpe this
# images = img_resized.reshape(-1, 28*28)
# images = images.float()
# outputs = model(images)
# _, predicted = torch.max(outputs.data, 1)
# print("Số được nhận biết: ", predicted[0].item())
# torch.save(model, 'model.ckpt')
# plt.show()
