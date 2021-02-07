from tkinter import *
import cv2
from PIL import Image,ImageTk
import requests
url='http://192.168.149.239:5000/video_feed'
def take_snapshot():
    ret=requests.get('http://192.168.149.239:5000/record')
    camera.release()
    cv2.destroyAllWindows()

def video_loop():
    success, img = camera.read()  # 从摄像头读取照片
    if success:
        cv2.waitKey(1000)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        root.after(1, video_loop)

camera = cv2.VideoCapture(url)    #摄像头

root = Tk()
root.title("get_face_img")

panel = Label(root)  # initialize image panel
panel.pack(padx=10, pady=10)
root.config(cursor="arrow")
btn = Button(root, text="get_photo", command=take_snapshot)
btn.pack(fill="both", expand=True, padx=10, pady=10)

video_loop()
root.mainloop()
camera.release()
cv2.destroyAllWindows()



