#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
# Load model
import cv2
import onnxruntime
import time
img_rows,img_cols=64,64

def random_crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0
    
    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift 
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/20.0
    else:
        dsteering = 0
    steering += dsteering
    
    return image,steering

ort_session = onnxruntime.InferenceSession("car_lap1.onnx")
#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)


start_time=time.time()
#registering event handler for the server
global show
@sio.on('telemetry')
def telemetry(sid, data):
    global show
    if data:
        
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
 
        # Compute steering angle of the car    
        
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # img=cv2.imread(BytesIO(base64.b64decode(data["image"])))
        # print(img.shape)
        
        image_array = np.asarray(image)
        image_array,_ = random_crop(image_array,rand=False)
        
        
        image_array=image_array/217.5-1.0
        
        
        pic = image_array.transpose(2,0,1).reshape(1,3,64,64).astype(np.float32)
        
      
   
 
        ort_inputs = {ort_session.get_inputs()[0].name:pic}
    
        ort_outs = ort_session.run(None, ort_inputs)
        print('result',ort_outs[0][0][0])
    #if ort_outs[0][0][0]>0.7:
        print(steering_angle)
        steering_angle=ort_outs[0][0][0]
        # round(ort_outs[0][0][0].astype(np.float64),2)
        
        
        
       # steering_angle = model.predict(image, preloaded=True)
        # Compute speed
        speed_target = 15 - abs(steering_angle) / 0.4 * 10
        #throttle = 0.2 - abs(steering_angle) / 0.4 * 0.15
        throttle = (speed_target - speed) * 0.1
        throttle = 1
        print(speed)
        
        print("network prediction -> (steering angle: {:.3f}, throttle: {:.3f})".format(steering_angle, throttle))

        send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('localhost', 4567)), app)
