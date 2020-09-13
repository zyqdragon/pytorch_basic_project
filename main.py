from models import ResNet34
from config import DefaultConfig
from data.dataset import DogCat
from torch.utils.data import Dataset, DataLoader
import torch as t
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from .models.ResNet34 import DogCat

opt=DefaultConfig()
print('----opt=',opt)
lr=opt.lr
print('---opt.train_data_root=',opt.train_data_root)

# step1: models
net=ResNet34.ResNet()
# print('----net=',net)
train_dataset = DogCat(opt.train_data_root, train=True)
val_dataset = DogCat(opt.train_data_root, train=False)

# step2: data set
train_dataloader=DataLoader(train_dataset,opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers)
val_dataloader=DataLoader(val_dataset,4,
                            shuffle=True,
                            num_workers=opt.num_workers)

# step3: target function and optimizer
criterion=t.nn.CrossEntropyLoss()
lr=opt.lr
# optimizer=t.optim.Adam(net.parameters(),lr=lr,weight_decay=opt.weight_decay)
optimizer=t.optim.SGD(net.parameters(),lr=lr,weight_decay=opt.weight_decay)

# step4: test the dataset and save the image samples #############################
# for ik in range(len(train_dataset)):
#     print('---ik=',ik,'----label=',train_dataset[ik][1],'----img=',train_dataset[ik][0])
# ik=109
# print('-----img_size=',pt1.shape)
# plt.imshow(pt1.numpy().transpose((1, 2, 0)))
# plt.savefig('./img_'+str(ik)+'_'+str(train_dataset[ik][1])+'.jpg')
# pt1=train_dataset[ik][0]
# print('---ik=',ik,'----label=',train_dataset[ik][1])

# step5: summary the indexes;
net.cuda()
# step6: training
for epoch in range(opt.max_epoch):
    for ik, (data,label) in enumerate(train_dataloader):
        input=Variable(data)
        target=Variable(label)
        if opt.use_gpu:
            input=input.cuda()
            target=target.cuda()
        optimizer.zero_grad()
        score=net(input)
        loss=criterion(score,target)
        loss.backward()
        optimizer.step()
    # t.save(net, 'model'+str(epoch)+'.pth')
    t.save(net.state_dict(), 'params'+str(epoch)+'.pth')

    # validation of the model
    net.eval()
    for ik, (data, label) in enumerate(val_dataloader):
        val_input=Variable(data,volatile=True)
        val_label=Variable(label.long(),volatile=True)
        if opt.use_gpu:
            val_input=val_input.cuda()
            val_label=val_label.cuda()
        score=net(val_input)
    print('----------------------test------------------------------')
    print('----------epoch=',epoch,'----loss=',loss)
    print('----the probability is:', t.nn.functional.softmax(score)[:, 1].data.tolist())
    print('----score=', score, '---label=', label)
    net.train()

print('----net_pre=',net.pre)
print('----------END---------------')