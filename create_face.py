from PIL import Image,ImageDraw
import numpy as np
from facenet_pytorch import MTCNN
from matplotlib.pyplot import imshow

import cv2

import math
import os
from tqdm import tqdm
import pandas as pd
mtcnn = MTCNN(image_size=120,select_largest=False)





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






root='/home/xu/data/VGG-Face2/data/vggface2_train/train/'
nroot='/home/xu/train/'

# root='/home/xu/data/VGG-Face2/data/vggface2_test/test/'
# nroot='/home/xu/test/'





landmarks_csv=pd.read_csv('/home/xu//data/VGG-Face2/meta/bb_landmark/loose_landmark_train.csv').values

# landmarks_csv=pd.read_csv('/home/xu//data/VGG-Face2/meta/bb_landmark/loose_landmark_test.csv').values

for i in tqdm(range(len(landmarks_csv))):
  
    addr=root+landmarks_csv[i][0]+'.jpg'
  # if '/n000131/0344_02.jpg' in addr:
    naddr=nroot+landmarks_csv[i][0]+'.jpg'
    
    landmarks=np.array(landmarks_csv[i][1:]).reshape(5,2)
    



        
    if not os.path.exists(naddr[:-11]):
        os.makedirs(naddr[:-11])
   

    image_array=cv2.imread(addr)

    
        

           
    if filter_face(1.5,landmarks) and image_array.shape[0]>120 and image_array.shape[1]>120:
      try:
        aligned_face, eye_center, angle = align_face(image_array=image_array, landmarks=landmarks)

        rotated_landmarks=rotate_landmarks(landmarks, eye_center, angle, aligned_face.shape[0])

        cropped_img=corp_face(aligned_face, rotated_landmarks)
        
       
        if cv2.Laplacian(cropped_img, cv2.CV_64F).var()>150:
        
            cv2.imwrite(naddr,cropped_img)
      except:
          pass
    # pic=image_array.copy()
    # for land in landmarks:
    #     x,y=tuple(land)
    #     cv2.circle(pic,(int(x),int(y)),5,(0,255,255),0)
    
    else:
        pass



















