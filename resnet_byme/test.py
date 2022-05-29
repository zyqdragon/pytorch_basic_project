from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import os
import cv2
# from linformer import Linformer
# from vit_pytorch.efficient import ViT
# from resnet32 import ResNet
from model_resnet import ResNet

# device = 'cuda'
device = 'cpu'

model=ResNet().to(device)

if __name__ == '__main__':
   input_size = 224
   model=torch.load('./models/model_resnet.pth')
   #  model=torch.load('./models/vit_model_linear.pth')
   transform_valid = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()])
   #  img_list = os.listdir('./data/train')
   #  for filename in img_list:
   #     img = Image.open('./data/train/'+filename)
   #     img_ = transform_valid(img).unsqueeze(0) #拓展维度
   #     preds = model(img_.to('cuda:0')) # (1, 1000)
   #     kname, indices = torch.max(preds,1)
   #     print("-------img_name=",filename,"----kind=",indices)
   img = Image.open('./data/train/cat.5.jpg')
   img = Image.open('./data/train/dog.1.jpg')
   img.save("result.jpg") # save images based on PIL library
   print("----img=",img.size)
   img_ = transform_valid(img).unsqueeze(0) #拓展维度
   print("----img_shape=",img_.shape)
   img2=transform_valid(img)
   print("----img2_shape=",img2.shape)
   preds = model(img_.to('cuda:0')) # (1, 1000)
   print("---**--------preds=",preds)

   # # save the images after be transformed by cv2 library
   # array1=img2.numpy()#将tensor数据转为numpy数据
   # maxValue=array1.max()
   # array1=array1*255/maxValue#normalize，将图像数据扩展到[0,255]
   # mat=np.uint8(array1)#float32-->uint8
   # print('mat_shape:',mat.shape)#mat_shape: (3, 982, 814)
   # mat=mat.transpose(1,2,0)#mat_shape: (982, 814，3)
   # cv2.imwrite("imgnew.jpg",mat)
