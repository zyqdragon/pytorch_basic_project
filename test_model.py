from models import ResNet34
from PIL import Image
import torch as t
from torch.autograd import Variable
import matplotlib.pyplot as plt

model=ResNet34.ResNet()
model.load_state_dict(t.load('params99.pth'))
# print('-----model=',model)
model.eval()

img_data = Image.open('/home/cat_and_dog/data/test_imgs/s0009_cat.jpg')
img_data.save('test.png')
print('--img_data=',img_data)

import matplotlib
im = matplotlib.image.imread('/home/cat_and_dog/data/test_imgs/s0009_cat.jpg')
print('---img=',im)
