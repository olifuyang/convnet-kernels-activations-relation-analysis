import pickle
import torch
from torch.nn.parameter import Parameter
from PIL import Image
from torchvision.models import resnet34,resnet50,resnet152,ResNet34_Weights,ResNet50_Weights,ResNet152_Weights
import json
from pathlib import Path
from nltk.tokenize import word_tokenize

with open('C:/Users/lenovo/repos/my/imagenet-simple-labels/imagenet-simple-labels.json') as f:
    labels = json.load(f)
def class_id_to_label(i):
    return labels[i]

mpltllist=['rsn.layer1[0].conv1',
 'rsn.layer1[0].conv2',
 'rsn.layer1[0].conv3',
 'rsn.layer1[1].conv1',
 'rsn.layer1[1].conv2',
 'rsn.layer1[1].conv3',
 'rsn.layer1[2].conv1',
 'rsn.layer1[2].conv2',
 'rsn.layer1[2].conv3',
 'rsn.layer2[0].conv1',
 'rsn.layer2[0].conv2',
 'rsn.layer2[0].conv3',
 'rsn.layer2[1].conv1',
 'rsn.layer2[1].conv2',
 'rsn.layer2[1].conv3',
 'rsn.layer2[2].conv1',
 'rsn.layer2[2].conv2',
 'rsn.layer2[2].conv3',
 'rsn.layer2[3].conv1',
 'rsn.layer2[3].conv2',
 'rsn.layer2[3].conv3',
 'rsn.layer3[0].conv1',
 'rsn.layer3[0].conv2',
 'rsn.layer3[0].conv3',
 'rsn.layer3[1].conv1',
 'rsn.layer3[1].conv2',
 'rsn.layer3[1].conv3',
 'rsn.layer3[2].conv1',
 'rsn.layer3[2].conv2',
 'rsn.layer3[2].conv3',
 'rsn.layer3[3].conv1',
 'rsn.layer3[3].conv2',
 'rsn.layer3[3].conv3',
 'rsn.layer3[4].conv1',
 'rsn.layer3[4].conv2',
 'rsn.layer3[4].conv3',
 'rsn.layer3[5].conv1',
 'rsn.layer3[5].conv2',
 'rsn.layer3[5].conv3',
 'rsn.layer4[0].conv1',
 'rsn.layer4[0].conv2',
 'rsn.layer4[0].conv3',
 'rsn.layer4[1].conv1',
 'rsn.layer4[1].conv2',
 'rsn.layer4[1].conv3',
 'rsn.layer4[2].conv1',
 'rsn.layer4[2].conv2',
 'rsn.layer4[2].conv3',
 'rsn.avgpool',
 'rsn.fc']

reallist=['rsn.conv1',
          'rsn.layer1[0].conv1',
          'rsn.layer1[0].conv2',
          'rsn.layer1[0].downsample[0]',
          'rsn.layer1[1].conv1',
          'rsn.layer1[1].conv2',
          'rsn.layer1[1].conv3',
          'rsn.layer1[2].conv1',
          'rsn.layer1[2].conv2',
          'rsn.layer1[2].conv3',
          'rsn.layer2[0].conv1',
          'rsn.layer2[0].conv2',
          'rsn.layer2[0].downsample[0]',
          'rsn.layer2[1].conv1',
          'rsn.layer2[1].conv2',
          'rsn.layer2[1].conv3',
          'rsn.layer2[2].conv1',
          'rsn.layer2[2].conv2',
          'rsn.layer2[2].conv3',
          'rsn.layer2[3].conv1',
          'rsn.layer2[3].conv2',
          'rsn.layer2[3].conv3',
          'rsn.layer3[0].conv1',
          'rsn.layer3[0].conv2',
          'rsn.layer3[0].downsample[0]',
          'rsn.layer3[1].conv1',
          'rsn.layer3[1].conv2',
          'rsn.layer3[1].conv3',
          'rsn.layer3[2].conv1',
          'rsn.layer3[2].conv2',
          'rsn.layer3[2].conv3',
          'rsn.layer3[3].conv1',
          'rsn.layer3[3].conv2',
          'rsn.layer3[3].conv3',
          'rsn.layer3[4].conv1',
          'rsn.layer3[4].conv2',
          'rsn.layer3[4].conv3',
          'rsn.layer3[5].conv1',
          'rsn.layer3[5].conv2',
          'rsn.layer3[5].conv3',
          'rsn.layer4[0].conv1',
          'rsn.layer4[0].conv2',
          'rsn.layer4[0].downsample[0]',
          'rsn.layer4[1].conv1',
          'rsn.layer4[1].conv2',
          'rsn.layer4[1].conv3',
          'rsn.layer4[2].conv1',
          'rsn.layer4[2].conv2',
          'rsn.layer4[2].conv3'
         ]

weights = ResNet50_Weights.DEFAULT
rsn = resnet50(weights=weights)
rsn.eval()

preprocess = weights.transforms()

openpkl1=open(r'C:\Users\lenovo\repos\my\insight\classspecdata.pkl', 'rb')
myorgdict = pickle.load(openpkl1)
openpkl1.close()

openpkl2=open(r'C:\Users\lenovo\repos\my\insight\classdiffdata.pkl', 'rb')
mydict = pickle.load(openpkl2)
openpkl2.close()

idx=0
prerr=0
for lyer in mpltllist:
    masklst=[]
    err=0
    for num in mydict['hen'][lyer]:
        if num:
            masklst.append(0)
        else:
            masklst.append(1)
    masknext = torch.Tensor(masklst)
    eval(reallist[idx]).weight=Parameter(torch.mul(eval(reallist[idx]).weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext,1),1),1)))
    
    if prerr>80:
        print('End!!!')
        break

    imgpath=Path(r'E:\BaiduNetdiskDownload\ILSVRC2012\trainT')
    for eachdir in imgpath.iterdir():
        label=''
        ct=0
        for image in list(eachdir.glob('**/*.JPEG')):
            #print('Image:',eachdir,image)
            img=Image.open(image)
            if 'L' ==  img.getbands()[0]:
                continue
            T=torch.softmax(torch.Tensor(rsn(preprocess(img).unsqueeze(0))),1)
            for i in range(0,999):
                if T.max() == T.squeeze(0)[i]:
                    #print(i,"||",class_id_to_label(i))
                    label=class_id_to_label(i)
                    ct+=1
                    break
            if label!='hen':
                err+=1
            if ct%100==0:
                print('#x#',idx,'have ',err,'error!!!')
                prerr=err
                break
    idx+=1


#masklst1=[]
#masklst2=[]
#masklst3=[]

#intersection=list(set(torch.argwhere(mydict['cock']['rsn.layer4[2].conv3']).tolist()).intersection(
#set(torch.argwhere(myorgdict['cock']['rsn.layer4[2].conv3']).tolist())))

"""
for num in mydict['cock']['rsn.layer1[0].conv1']:
    if num:
        masklst1.append(0)
    else:
        masklst1.append(1)
masknext1 = torch.Tensor(masklst1)

for num in mydict['cock']['rsn.layer1[0].conv2']:
    if num:
        masklst2.append(0)
    else:
        masklst2.append(1)
masknext2 = torch.Tensor(masklst2)


for num in mydict['cock']['rsn.layer4[2].conv3']:
    #if num:
        masklst1.append(0)
    #else:
    #    masklst1.append(1)
masknext1 = torch.Tensor(masklst1)

#[[162], [187], [197], [309], [406], [446], [447]]
#masknext1.index_fill_(0,torch.tensor([162]),0)
#masknext1.index_fill_(0,torch.tensor([187]),0)
#masknext1.index_fill_(0,torch.tensor([197]),0)
#masknext1.index_fill_(0,torch.tensor([309]),0)
#masknext1.index_fill_(0,torch.tensor([406]),0)
#masknext1.index_fill_(0,torch.tensor([446]),0)
#masknext1.index_fill_(0,torch.tensor([447]),0)

"""
"""
for num in mydict['cock']['rsn.layer4[2].conv2']:
    if num:
        masklst2.append(0)
    else:
        masklst2.append(1)
masknext2 = torch.Tensor(masklst2)

for num in mydict['cock']['rsn.layer4[0].conv3']:
    if num:
        masklst3.append(0)
    else:
        masklst3.append(1)
masknext3 = torch.Tensor(masklst3)
"""
#masknext1 = mydict['cock']['rsn.layer1[0].conv1']
#rsn.layer4[0].conv2.weight = Parameter(torch.mul(rsn.layer4[0].conv2.weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext3,1),1),1)))
#rsn.layer4[2].conv1.weight = Parameter(torch.mul(rsn.layer4[2].conv1.weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext2,1),1),1)))
#rsn.layer4[2].conv2.weight = Parameter(torch.mul(rsn.layer4[2].conv2.weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext1,1),1),1)))
#rsn.conv1.weight=Parameter(torch.mul(rsn.conv1.weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext1,1),1),1)))
#rsn.layer1[0].conv1.weight=Parameter(torch.mul(rsn.layer1[0].conv1.weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext2,1),1),1)))
#rslt=torch.broadcast_tensors(rsn.conv1.weight,torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(masknext,1),1),1))
#rsn.conv1.weight = Parameter(torch.mul(rsn.conv1.weight,rslt[1]))
"""
imgpath=Path(r'E:\BaiduNetdiskDownload\ILSVRC2012\train2')

for eachdir in imgpath.iterdir():
    for image in list(eachdir.glob('**/*.JPEG')):
        print('Image:',eachdir,image)
        img=Image.open(image)
        if 'L' ==  img.getbands()[0]:
            continue
        T=torch.softmax(torch.Tensor(rsn(preprocess(img).unsqueeze(0))),1)
        for i in range(0,999):
            if T.max() == T.squeeze(0)[i]:
                print(i,"||",class_id_to_label(i))
"""