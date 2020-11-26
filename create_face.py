from facenet_pytorch import MTCNN
import math
from PIL import Image
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
device=torch.device('cuda:0')

def get_angle(landmarks):
    x1,y1,x2,y2=landmarks[0][3][0],landmarks[0][3][1],landmarks[0][4][0],landmarks[0][4][1]
    eye_center = (x1+x2)/2,(y1+y2)/2
    
    dx = x2-x1
    dy = y2-y1

    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    return RotateMatrix


mtcnn = MTCNN(image_size=120)


# face_tensor, prob = mtcnn(img, save_path='face2.png', return_prob=True)  # 返回的是检测到的人脸数据tensor 形状是N，3*160*160 # 尺寸不一定是160，之前的参数设置
# boxes, prob = mtcnn.detect(img)


root='/home/xu/data/test/'
nroot='/home/xu/py_work/face2/'
names=os.listdir(root)

for name in tqdm(names):
    pics=os.listdir(root+name)
    for pic in pics:
        img=Image.open(root+name+'/'+pic)
        
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        
        try:
          if boxes.shape[0]!=1 :
            pass
          else:
            
                Matrix=get_angle(landmarks)
                
                n_face = cv2.warpAffine(np.array(img), Matrix, (np.array(img).shape[0], np.array(img).shape[1])) 
                
                face_tensor, prob = mtcnn(n_face, save_path=nroot+name+'/'+pic, return_prob=True)
                
                
        except:
                print(root+name+'/'+pic+'  failed!')
    
    
