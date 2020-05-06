import os
import re
import cv2
import torch
import numpy as np
from torch import nn


def predict(X):
    device = torch.device("cpu")
    DROPOUT1 = 0.2
    DROPOUT2 = 0.6

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, (5, 5))  # output (n_examples, 16, 116, 156)
            self.convnorm1 = nn.BatchNorm2d(8)
            self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 58, 78)
            self.conv2 = nn.Conv2d(8, 16, (3, 3))  # output (n_examples, 64, 56, 76)
            self.convnorm2 = nn.BatchNorm2d(16)
            self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 28, 38)
            self.conv3 = nn.Conv2d(16, 32, (3, 3))  # output (n_examples, 256, 26, 36)
            self.convnorm3 = nn.BatchNorm2d(32)
            self.pool3 = nn.MaxPool2d((2, 2))  # output (n_examples, 48, 13, 18)
            self.conv4 = nn.Conv2d(32, 48, (3, 3))  # output (n_examples, 32, 11, 16)
            self.relectpad = nn.ReflectionPad2d((0, 0, 1, 0))  # 13 18
            self.convnorm4 = nn.BatchNorm2d(48)
            self.pool4 = nn.MaxPool2d((2, 2))  # output (n_examples, 64, 6, 9)
            self.linear1 = nn.Linear(48 * 6 * 8, 128)  # input will be flattened to (n_examples, 48*4*4)
            self.linear1_bn = nn.BatchNorm1d(128)
            self.drop1 = nn.Dropout(DROPOUT1)
            self.linear2 = nn.Linear(128, 64)
            self.linear2_bn = nn.BatchNorm1d(64)
            self.drop2 = nn.Dropout(DROPOUT2)
            self.linear3 = nn.Linear(64, 7)
            self.act = nn.ReLU(inplace=True)
            self.sigmoid = torch.sigmoid

        def forward(self, x):
            x = self.pool1(self.convnorm1(self.act(self.conv1(x.permute(0, 3, 1, 2)))))
            x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
            x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
            x = self.pool4(self.convnorm4(self.act(self.relectpad(self.conv4(x)))))
            x = self.drop1(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
            x = self.drop2(self.linear2_bn(self.act(self.linear2(x))))
            return self.sigmoid(self.linear3(x))

    # Read the images
    image_id = []
    for filename in X:
        if re.findall(r"\.png", filename):
            img = cv2.imread(filename)
            image_id += [(img, filename.strip(r"\.png"))]
    res = np.zeros([len(image_id), 120, 160, 3])
    # Resize the images
    for i in range(len(image_id)):
        res[i] = cv2.resize(image_id[i][0], (160, 120), interpolation=cv2.INTER_CUBIC)
    # Reshape the data
    x_test = torch.tensor(res).float().to(device)
    # Scale the data
    x_test = x_test / 255
    # Load the model
    model = CNN().to(device)
    PATH1 = 'model_yongchaoqiao91.pt'
    PATH2 = 'model_yongchaoqiao92.pt'
    PATH3 = 'model_yongchaoqiao93.pt'
    PATH4 = 'model_yongchaoqiao94.pt'
    PATH5 = 'model_yongchaoqiao95.pt'

    # Test the saved best model
    model.load_state_dict(torch.load(PATH1, map_location=device))
    model.eval()
    y_pred_1 = model(x_test).detach()
    model.load_state_dict(torch.load(PATH2, map_location=device))
    model.eval()
    y_pred_2 = model(x_test).detach()
    model.load_state_dict(torch.load(PATH3, map_location=device))
    model.eval()
    y_pred_3 = model(x_test).detach()
    model.load_state_dict(torch.load(PATH4, map_location=device))
    model.eval()
    y_pred_4 = model(x_test).detach()
    model.load_state_dict(torch.load(PATH5, map_location=device))
    model.eval()
    y_pred_5 = model(x_test).detach()
    # Get the predict
    y_test_pred = (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4 + y_pred_5) / 5
    return y_test_pred
