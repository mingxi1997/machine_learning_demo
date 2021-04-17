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

import onnxruntime

ort_session = onnxruntime.InferenceSession("/home/xu/py_work/car_lap1.onnx")
#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)



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
        
        
        pic=np.array(image).astype(np.float32)
       
        pic=pic[65:-25, :, :]
        pic=np.transpose(pic,(2,0,1))
        pic=pic.reshape(1,3,70,320)
        
        ort_inputs = {ort_session.get_inputs()[0].name:pic}
    
        ort_outs = ort_session.run(None, ort_inputs)
    #if ort_outs[0][0][0]>0.7:
    
        steering_angle=round(ort_outs[0][0][0].astype(np.float64),2)
        
        
        
       # steering_angle = model.predict(image, preloaded=True)
        # Compute speed
        speed_target = 25 - abs(steering_angle) / 0.4 * 10
        # throttle = 0.2 - abs(steering_angle) / 0.4 * 0.15
        throttle = (speed_target - speed) * 0.1
        throttle = 1
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
