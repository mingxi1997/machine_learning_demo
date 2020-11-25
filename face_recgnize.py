import numpy as np
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import math
from triplet_test import RES
import torch

import torchvision

import time
device=torch.device('cuda:0')
 
model=RES()

model.load_state_dict(torch.load('face_rec90.pt'))


mtcnn = MTCNN(image_size=120)

cap = cv2.VideoCapture(0)


def ajust(array,landmarks):
    x1,y1,x2,y2=landmarks[0][3][0],landmarks[0][3][1],landmarks[0][4][0],landmarks[0][4][1]
    
    
    eye_center = (x1+x2)/2,(y1+y2)/2
    
    dx = abs(x2-x1)
    dy = abs(y2-y1)

    angle = -math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵

    align_face = cv2.warpAffine(array, RotateMatrix, (array.shape[0], array.shape[1]))  # 进行放射变换，即旋转
    
    return align_face

me=torch.load('me.pt').cpu()
me2=torch.load('me2.pt')

min_v=2
while(True):
    s=time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()


    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    
    if type(boxes)==type(None):
        pass
    else:
   
        x,y,w,h=boxes[0]
    
        draw_1=cv2.rectangle(frame, (x,y), (w,h), (0,255,0), 2)

        cv2.imshow('frame',draw_1)
        img=Image.fromarray(frame)

    
    
    
        region = img.crop((x, y, w, h)).resize((120,120))
        
        
        
    
        r=np.array(region)
        
        
        
       
        
        trans=torchvision.transforms.ToTensor()
        
        emb=trans(r).unsqueeze(0).to(torch.float32)
        out=model(emb)
       
        
        # print(torch.dist(me,out))
        if torch.dist(me,out).item()<min_v:
            min_v=torch.dist(me,out).item()
            print('ok',min_v)
        if torch.dist(me,out).item()<1.5:
            print('xu hong')
        if torch.dist(me2,out).item()<1.5:
            print('xu qijun')
       
        
            
            
        e=time.time()
        # print(e-s)
    # Display the resulting frame
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
