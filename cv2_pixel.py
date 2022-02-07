


import cv2
import matplotlib.pyplot as plt

def plotImg(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        
cap=cv2.VideoCapture('/home/xu/Downloads/数量任务/胜利村南1.MOV')
while True:
 #从摄像头读取图片
    ret,img=cap.read()
    
    # img=img[:-300]
# img = cv2.imread('test.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 131, 15)
    plotImg(binary_img)
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
    # first box is the background
    boxes = boxes[1:]
    filtered_boxes = []
    for x,y,w,h,pixels in boxes:
        # if pixels < 10000 and h < 200 and w < 200 and h > 10 and w > 10:
        if w<50 and h<50 and w>4 and h>4:
            filtered_boxes.append((x,y,w,h))
    
    for x,y,w,h in filtered_boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
    print(boxes.shape[0])
    # plotImg(img)
    
    cv2.putText(img, 'nums : '+str(boxes.shape[0]), (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 3)
    
    cv2.imshow("img",img)    
    
    cv2.waitKey(1)
