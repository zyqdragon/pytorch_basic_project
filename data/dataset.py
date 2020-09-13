import os, sys
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    def __init__(self,img_folder,transforms=None,train=True,test=False):
         self.test=test
         imgs=[os.path.join(img_folder,img) for img in os.listdir(img_folder)]
         # imgs=os.listdir(img_folder)
         imgs.sort()
         # print('----imgs=',imgs)
         imgs_num=len(imgs)
         print('----imgs=',imgs_num)

         # divide training,verification,test
         if self.test:
             self.imgs=imgs
         elif train:
             self.imgs=imgs[:int(1.0*imgs_num)]
         else:
             self.imgs=imgs[int(0.95*imgs_num):]

         if transforms is None:
             normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

         if self.test or not train:
             self.transforms=T.Compose([
                 #T.Scale(224),
                 T.Resize(224),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 normalize
             ])
         else:
             self.transforms=T.Compose([
                 #T.Scale(256),
                 T.Resize(256),
                 T.RandomResizedCrop(224),
                 T.RandomHorizontalFlip(),
                 T.ToTensor(),
                 normalize
             ])
    def __getitem__(self,index):
        img_path=self.imgs[index]
        # if self.test:
        #     label=int(self.imgs[index].split('.')[-2].split('/')[-1])
        # else:
        #     label=1 if 'dog' in img_path.split('/')[-1] else 0
        if img_path[-7:-4]=='cat':
            # print('----img_path=',img_path)
            label=0
        elif img_path[-7:-4]=='dog':
            # print('----img_path=', img_path)
            label=1
        else:
            print('----img_path[-7:-4]=', img_path[-7:-4])
            print('---wrong name of images-----')
            sys.exit(0)

        data= Image.open(img_path)
        data=self.transforms(data)
        return data,label

    def __len__(self):
        return len(self.imgs)

# train_dataset=DogCat('../dataset/train_imgs',train=True)
# train_dataset=DogCat('./train_imgs',train=True)
# trainloader=DataLoader(train_dataset,
#                        batch_size=opt.batch_size,
#                        shuffle=True,
#                        num_workers=opt.num_workers)
# print(train_dataset[1])
# print(train_dataset.imgs[0])
# for ik in range(len(train_dataset)):
#     print('---ik=',ik,'----label=',train_dataset[ik][1])
# print('-------------end of dataset------------')
