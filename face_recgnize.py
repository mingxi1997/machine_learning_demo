from PIL import Image,ImageDraw
import numpy as np
from facenet_pytorch import MTCNN
from matplotlib.pyplot import imshow

import cv2

import math
import os
from tqdm import tqdm
import torchvision
import torch
import torch.nn as nn
import torchvision.models as models

mtcnn = MTCNN(image_size=120,select_largest=False)
device=torch.device('cuda:0')

class RES(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.fc=nn.Linear(1000,128)
    def forward(self,x):
        out=self.model(x)
        
        out=self.fc(out)
        out=torch.nn.functional.normalize(out)
        # out=torch.normalize(out)
        return out    

# model=RES()
# model.load_state_dict(torch.load('face_net_res18.pt'))
# model=model.to(device)

from inceptv1 import InceptionResnetV1
model=InceptionResnetV1()
# model.load_state_dict(torch.load('face_net_c.pt'))
model.load_state_dict(torch.load('face_net_sgd.pt'))
model=model.to(device)



# from facenet_pytorch import MTCNN, InceptionResnetV1
# model = InceptionResnetV1(pretrained='vggface2')
# model=model.to(device)



def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    
    rotated_landmarks = [rotate(origin=eye_center, point=landmark, angle=angle, row=row) for landmark in landmarks]
            
    return rotated_landmarks


def align_face(image_array, landmarks):
    x1,y1,x2,y2=landmarks[0][0],landmarks[0][1],landmarks[1][0],landmarks[1][1]
    eye_center = (x1+x2)/2,(y1+y2)/2
    
    dx = x2-x1
    dy = y2-y1
    # at the eye_center, rotate the image by the angle
    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    
    rotated_img = cv2.warpAffine(image_array, RotateMatrix , (image_array.shape[1], image_array.shape[0])) 
    return rotated_img, eye_center, angle


def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks[0]),
                                    np.array(landmarks[1])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks[3]),
                                    np.array(landmarks[4])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks[3], axis=0)[0]
    x_max = np.max(landmarks[3], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top



def corp_face(image_array, landmarks):
    eye_landmark = np.stack([np.array(landmarks[0]),
                                    np.array(landmarks[1])],axis=0)
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.stack([np.array(landmarks[3]),
                                    np.array(landmarks[4])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    
    
    x_min = np.min(landmarks, axis=0)[0]
    x_max = np.max(landmarks, axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom)).resize((120,120))
    cropped_img = np.array(cropped_img)
    return cropped_img
def square(boxes):
    x1,y1,x2,y2=boxes
    return (x2-x1)*(y2-y1)
    

def filter_face(criterion,landmarks):

  
    l=landmarks[0][0]
    r=landmarks[1][0]
    m=landmarks[2][0]
    if r==m:
        return False
    else :
        ratio=(m-l)/(r-m)

        if ratio>criterion or ratio<1/criterion:
 
            return False
        else:
            return True



def identy(box):
    x1,y1,x2,y2=box
    x_center=(x1+x2)/2
    y_center=(y1+y2)/2
    
    s=(y2-y1)*(x2-x1)
    
    return np.array([x_center,y_center,s])

def check(box):
    re=True
    for x in box:
        if x<1/1.5 or x>1.5:
            re=False
            break
    return re


def near_judge(box1,record):
    
    box1=identy(box1)

    result=False
    box=[0,0,0,0]
    name='none'
    for n,box2 in record.items():
        if check(box1/box2):
            
            
                box=box1
                name=n
                result=True
                break
    return result,name,box





cap = cv2.VideoCapture(0)
model.eval()


pp1=torch.load('sgd1.pt')

p1=torch.load('11.pt')



# p2=torch.load('2.pt')
p2=torch.load('22.pt')
p3=torch.load('33.pt')

p4=torch.load('4.pt')
t=[]
x=[]
y=[]
box_record={}

threshhold=0.01

yesterday=[]
c=0
while True:
  
    # Capture frame-by-frame
    
    
    
    
    
    ret, frame = cap.read()


    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
    
    
    
    if type(boxes)==type(None):
        box_record={}
        cv2.imshow('1',frame)
        cv2.waitKey(1)       
           
        pass
    else:
                     
            
        for i in range(len(boxes)):
            
            x1,y1,x2,y2=boxes[i]
           
            
            
            if square(boxes[i])<1440:
                pass
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

                landmark=landmarks[i]
                
                
                
                
                ret,name,box=near_judge(boxes[i],box_record)
                if ret:
                    cv2.putText(frame, name, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 
                    print('from record')
                
                
                
                elif filter_face(1.5,landmark):
                
                
                  

                  aligned_face, eye_center, angle = align_face(image_array=frame, landmarks=landmark)

                  rotated_landmarks=rotate_landmarks(landmark, eye_center, angle, aligned_face.shape[0])

                  cropped_img=corp_face(aligned_face, rotated_landmarks)
                
                  trans=torchvision.transforms.ToTensor()
                 
                  emb=trans(cropped_img).unsqueeze(0).to(torch.float32).to(device)
                
                  out=model(emb)
                
             
                  print(torch.dist(out,pp1))
               
                  if torch.dist(out,p1)<0.5:
                      # print(torch.dist(out,p1),'111')
                      
                      
               
                      cv2.putText(frame, 'xu hong', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 
                      
                     
                      
                      box_record['xu hong']=identy(boxes[i])
                      
                  if torch.dist(out,p2)<0.5 :
                     
                      cv2.putText(frame, 'caijikai', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 
                      
                      box_record['caijikai']=identy(boxes[i])
                  if torch.dist(out,p3)<0.5 :
                     
                      cv2.putText(frame, 'hou', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 
                      
                      box_record['hou']=identy(boxes[i])
                      
                      
            
                      
                      
                      
                      
                      # print(torch.dist(out,p1),'111')
                #       # print(torch.dist(out,p2),'222')
                #      cv2.putText(frame, 'caoreping', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 

                #     elif torch.dist(out,p2)<0.5 :
                #       print('222')
                #       cv2.putText(frame, 'xuqijun', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 
                      
                #     elif torch.dist(out,p3)<0.5:
                #       print('333')
                      
                #     elif torch.dist(out,p4)<0.5:
                #       print('333')
                #       cv2.putText(frame, 'xu siwei', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA) 
                #     else:
                #       cv2.putText(frame, 'can not rec', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,255,255), 2, cv2.LINE_AA)
        
            
        cv2.imshow('1',frame)
        cv2.waitKey(1)       
           
                
                
                # else:
             
                    
                 
                  # x.append(torch.dist(out,p1))
                  # if len(x)>30 and sum(x)/len(x)<0.001 and sum(x)/len(x)<sum(y)/len(y):
                  #   print('xu')
                  #   x=[]
                
              
 
                  # y.append(torch.dist(out,p2))
                  # if len(y)>30 and sum(y)/len(y)<0.001 and sum(x)/len(x)>sum(y)/len(y):
                  #   print('xuqijun')
                  #   y=[]
                
                
                   
 

            # cv2.imwrite('test2.jpg',cropped_img)


















