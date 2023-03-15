import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


input_size = 784
hidden_sizes = [128, 64]
output_size = 10


model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9) 
criterion = nn.NLLLoss() #Negative log loss function for n classes


model.load_state_dict(torch.load('/media/laven/Linux Storage/VSC/Digit_Detector.pt'),map_location='cpu')
model.eval()
load_from_sys = True

if load_from_sys:
    hsv_value = np.load('hsv_value.npy')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('cannot open camera')
    exit()

cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.uint8)

canvas = None

x1 = 0
y1 = 0

noise_thresh = 800

while True:
    ret, frame = cap.read()
    if not ret:
        print('cant recieve video')
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if load_from_sys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]

    mask = cv2.inRange(hsv, lower_range, upper_range)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)


    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours and cv2.contourArea(max(contours, key= cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            canvas = cv2.line(canvas, (x1, y1), (x2, y2), [130, 255, 255], 4)

        x1, y1 = x2, y2

    else:
        x1, y1 = 0, 0

    frame = cv2.add(frame, canvas)

    stacked = np.hstack((canvas, frame))
    cv2.imshow('screen_pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))



    if cv2.waitKey(1) == ord('p'):
        blw = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        blw = cv2.resize(blw, (28, 28))
        blw = torch.Tensor(blw)
        blw = torch.Tensor.view(blw, (1, 1, 28, 28))
        y, x = model(blw)
        print(torch.max(y, 1)[1].data.squeeze())
        break


cv2.destroyAllWindows()
cap.release()