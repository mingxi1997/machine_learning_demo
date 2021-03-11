import numpy as np
from facenet_pytorch import MTCNN
from matplotlib.pyplot import imshow

import cv2

import math
import os

import torchvision
import torch
import torch.nn as nn
import torchvision.models as models

mtcnn = MTCNN(image_size=120,select_largest=False)




from facenet_pytorch import MTCNN, InceptionResnetV1
model = InceptionResnetV1(pretrained='vggface2')
model.eval()
# model.load_state_dict(torch.load('20180402-114759-vggface2.pt')).eval()

x = torch.randn(1, 3, 120, 120)
torch_out = model(x)
torch.onnx.export(model,
                  x,
                  "test_to_onnx.onnx",
                  export_params = True,
                  opset_version=9,
                  do_constant_folding=True,
                  input_names = ['x'],
                  output_names = ['y']
                  )

