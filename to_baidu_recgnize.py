# encoding:utf-8
import os
import cv2
import base64
import requests
video_src='test.mp4'
cap = cv2.VideoCapture(video_src)
with open('result.txt','r') as r:
    contents=r.readlines()
i=0

def get_name(frame):
    cv2.imwrite('t.jpg',frame)
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/animal"

    with open('t.jpg', 'rb') as f:
        img = base64.b64encode(f.read())
        params = {"image": img}
        access_token = '24.9af227fbd2327b23a1921d37dcd829ad.2592000.1641623783.282335-25315317'
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        result = response.json()

        bird_name = result['result'][0]['name']
        return bird_name









record=''

while True:

    
    
    ret, frame = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
    
    if contents[i].split('\t')[1]=='0\n':
        pass
    else:
        name=get_name(frame)
        if name!='非动物':
            record=str(i+1)+'\t'+name+'\n'
            with open('record.txt','a')as f:
                f.write(record)
            print(name)
    label=contents[i]
    units=label.split('\t')
    count=units[0]
    num = int(units[1])
    # boxes=[]
    if num!=0:
        # print(label)
        cou=0
        for n in range(num):
            cou+=1
            path='crop_img/{}'.format(count)
            if not os.path.exists(path):
                os.mkdir(path)
            box=units[2+n*5:7+n*5]
            # boxes.append(units[2+n*5:7+n*5])
            score=float(box[0])
            x1=int(box[1])
            y1=int(box[2])
            x2=int(box[3])
            y2=int(box[4])
            # crop=frame[y1:y2, x1:x2]
            # cv2.imwrite('crop_img/{}/{}.jpg'.format(count,cou),crop)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
           
            # cv2.putText(frame, '{}'.format(name),
            #             (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 0), 1)
        # print(boxes)

    cv2.putText(frame, '{}'.format(i),
                (300, 300 ), cv2.FONT_HERSHEY_SIMPLEX, 5,
                (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    
    
    if cv2.waitKey(50)==ord('q'):
        cv2.imwrite('frame.jpg', frame)
        cv2.waitKey(5000)

    i+=1
