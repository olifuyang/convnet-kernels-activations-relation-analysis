import pickle
import torch
from torch.nn.parameter import Parameter
from PIL import Image
from torchvision.models import resnet34,resnet50,resnet152,ResNet34_Weights,ResNet50_Weights,ResNet152_Weights
import json
from pathlib import Path
from collections import Counter

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

mpltllist=['rsn.layer1[0].conv1', 
 'rsn.layer1[0].conv2', 
 'rsn.layer1[0].conv3', 
 'rsn.layer1[0].bn3', 
 'rsn.layer1[0].downsample[1].BatchNorm2d', 
 'rsn.layer1[1].conv1', 
 'rsn.layer1[1].conv2', 
 'rsn.layer1[1].conv3', 
 'rsn.layer1[2].conv1', 
 'rsn.layer1[2].conv2', 
 'rsn.layer1[2].conv3', 
 'rsn.layer2[0].conv1', 
 'rsn.layer2[0].conv2', 
 'rsn.layer2[0].conv3', 
 'rsn.layer2[0].bn3', 
 'rsn.layer2[0].downsample[1].BatchNorm2d', 
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
 'rsn.layer3[0].bn3', 
 'rsn.layer3[0].downsample[1].BatchNorm2d', 
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
 'rsn.layer4[0].bn3', 
 'rsn.layer4[0].downsample[1].BatchNorm2d', 
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
          'rsn.layer1[0].conv3',
          'rsn.layer1[0].downsample[0]',
          'rsn.layer1[0].afterplus',
          'rsn.layer1[1].conv1',
          'rsn.layer1[1].conv2',
          'rsn.layer1[1].conv3',
          'rsn.layer1[2].conv1',
          'rsn.layer1[2].conv2',
          'rsn.layer1[2].conv3',
          'rsn.layer2[0].conv1',
          'rsn.layer2[0].conv2',
          'rsn.layer2[0].conv3',
          'rsn.layer2[0].downsample[0]',
          'rsn.layer2[0].afterplus',
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
          'rsn.layer3[0].downsample[0]',
          'rsn.layer3[0].afterplus',
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
          'rsn.layer4[0].downsample[0]',
          'rsn.layer4[0].afterplus',
          'rsn.layer4[1].conv1',
          'rsn.layer4[1].conv2',
          'rsn.layer4[1].conv3',
          'rsn.layer4[2].conv1',
          'rsn.layer4[2].conv2',
          'rsn.layer4[2].conv3'
         ]

#keys=myorgdict.keys()
for key in myorgdict.keys():
    idx=1
    while idx < 57:
            if 'downsample' in mpltllist[idx]:
                for atv in torch.nonzero(myorgdict[key][mpltllist[idx]]): 
                    dikt=sorted(Counter(torch.flatten(torch.topk(torch.mul(eval(reallist[idx]).weight[int(atv)],torch.reshape(myorgdict[key][mpltllist[idx-4]],(int(myorgdict[key][mpltllist[idx-4]].size(dim=0)),1,1))),5,0)[1]).tolist()).items(),key=lambda item: item[1],reverse=True)
                    print(key,reallist[idx]+"'s Kernel","#"+str(int(atv[0])),"related to",reallist[idx-4]+"'s kernels:#",dikt)
                idx+=2
            else:
                for atv in torch.nonzero(myorgdict[key][mpltllist[idx]]): 
                    dikt=sorted(Counter(torch.flatten(torch.topk(torch.mul(eval(reallist[idx]).weight[int(atv)],torch.reshape(myorgdict[key][mpltllist[idx-1]],(int(myorgdict[key][mpltllist[idx-1]].size(dim=0)),1,1))),5,0)[1]).tolist()).items(),key=lambda item: item[1],reverse=True)
                    print(key,reallist[idx]+"'s Kernel","#"+str(int(atv[0])),"related to",reallist[idx-1]+"'s kernels:#",dikt)
                idx+=1