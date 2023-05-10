import pickle
import torch
from pathlib import Path
from collections import OrderedDict

"""
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
"""

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

clsagrdict=OrderedDict()
intersect={}
clsagrdictclssp=OrderedDict()

pklpath=Path(r'C:\Users\lenovo\repos\my\pklnew')
for eachpkl in list(pklpath.glob('**/*.pkl')):
    print(eachpkl)
    label=''.join(x for x in Path(eachpkl).stem if x.isalpha())
    VLU={}
    openpkl=open(eachpkl, 'rb')
    mylist = pickle.load(openpkl)
    openpkl.close()
    for anlz in mpltllist:
        if 'layer' in anlz or 'avgpool' in anlz:
            lenth=len(mylist)
            VLU[anlz]=torch.zeros_like((mylist)[0][anlz])
            for a in range(lenth):
                VLU[anlz]+=mylist[a][anlz]
            VLU[anlz]=torch.floor(VLU[anlz]/lenth)
            print('XXXX',anlz,mylist[a][anlz].shape,int(torch.count_nonzero(VLU[anlz])))
    clsagrdict[label]=VLU

outputfilename='classspecdata'+'.pkl'
output = open(outputfilename, 'wb')
pickle.dump(clsagrdict, output)
output.close()

for layer in mpltllist:
    if 'layer' in layer or 'avgpool' in layer:
        intersect[layer]=torch.zeros_like(clsagrdict[list(clsagrdict.keys())[0]][layer])
        for eachcls in clsagrdict.keys():
            intersect[layer]+=clsagrdict[eachcls][layer]
        intersect[layer]=torch.floor(intersect[layer]/len(clsagrdict.keys()))
        #print('XXXX',layer,clsagrdict[list(clsagrdict.keys())[0]][layer].shape,int(torch.count_nonzero(intersect[layer])))

for eachcls in clsagrdict.keys():
    clsagrdictclssp[eachcls]={}
    for layer in mpltllist:
        if 'layer' in layer or 'avgpool' in layer:
            clsagrdictclssp[eachcls][layer]=clsagrdict[eachcls][layer]-intersect[layer]
            print(eachcls,layer,clsagrdictclssp[eachcls][layer].shape,int(torch.count_nonzero(clsagrdictclssp[eachcls][layer])))
            print(torch.argwhere(clsagrdictclssp[eachcls][layer]))

outputfilename='classdiffdata'+'.pkl'
output = open(outputfilename, 'wb')
pickle.dump(clsagrdictclssp, output)
output.close()



       


