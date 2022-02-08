import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def plotImg(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        
cap = cv2.VideoCapture('/home/hw/project/count_number/城头乡九分厂码头南2.MOV')
w = cap.get(3)
h = cap.get(4)
frames = int(cap.get(7))-6
frame_rate = cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/hw/project/count_number/out.avi',fourcc,frame_rate,(int(w),int(h)))
time_line = []
nums = []
id = 0
while True and id < frames:
    ret,img=cap.read()    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 100, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 15,10)

    # 检测水面线
    # min_length = 100
    # max_gap = 55
    # lines = cv2.HoughLinesP(binary_img,1,np.pi/180,200,min_length,max_gap)
    # y = []
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         y.append(y1)
    #         y.append(y2)
    #         # cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    # y_mean = np.mean(y)

    # 腐蚀和膨胀操作，也可以用开运算
    # erode_img = cv2.erode(binary_img,None,iterations=1)
    # closing = cv2.morphologyEx(erode_img, cv2.MORPH_CLOSE, None)    
    # gradient = cv2.morphologyEx(closing, cv2.MORPH_GRADIENT, None)
    # dilate_img = cv2.dilate(gradient,None,iterations=1)
        
    
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
    
    # first box is the background
    boxes = boxes[1:]
    filtered_boxes = []
    for x,y,w,h,pixels in boxes:
        if w<50 and h<50 and w>1 and h>1 and w*h<400 and 0.5<w/h<2 and pixels<100:
        # if w<50 and h<50 and w>1 and h>1 and w*h<400 and 0.5<w/h<2 and y<y_mean:
             filtered_boxes.append((x,y,w,h))

    
    for x,y,w,h in filtered_boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),1)
        
    print(str(id), boxes.shape[0],cap.get(0))

    time_line.append(cap.get(0)/1000)
    nums.append(boxes.shape[0])
   
    cv2.putText(img, ' frames:'+ str(id+1)+'   nums : '+str(boxes.shape[0]), (35,50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)
    out.write(img)
    cv2.imshow("img",img)    
    kk = cv2.waitKey(1) 
    if kk == ord('q'):
        break
    id +=1 
# dataframe = pd.DataFrame({'frame':time_line,'nums':nums})
# dataframe.to_csv(r"test.csv",sep=',')
cv2.destroyAllWindows()    
# plt.plot(time_line,nums)

# # pyecharts scatter
# import pyecharts.options as opts
# from pyecharts.charts import Scatter

# c = (
#      Scatter(init_opts = opts.InitOpts(width='900px',height='600px'))
#      .add_xaxis(xaxis_data=time_line)
#      .add_yaxis(
#          series_name='',
#          y_axis=nums,
#          symbol_size=10,
#          symbol=None,
#          is_selected=True,
#          color='#00CCFF',
#          label_opts=opts.LabelOpts(is_show=False),
#          )
#      .set_series_opts()
#      .set_global_opts(
#          xaxis_opts=opts.AxisOpts(
#              name = 'time',
#              name_location='center',
#              name_gap=15,
#              type_='value',
#              splitline_opts=opts.SplitLineOpts(is_show=True)
#              ),
#          yaxis_opts=opts.AxisOpts(
#              name='numbers',
#              type_='value',
#              axistick_opts=opts.AxisTickOpts(is_show=True),
#              splitline_opts=opts.SplitLineOpts(is_show=True),
#              ),
#          tooltip_opts=opts.TooltipOpts(is_show=False),
#          )
#      .render("/home/hw/project/count_number/count_number.html")
#      )

