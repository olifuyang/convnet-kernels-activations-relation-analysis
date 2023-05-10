#for activation analysis
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

