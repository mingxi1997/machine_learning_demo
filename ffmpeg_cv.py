from os import write
import subprocess as sp
import cv2 

class ffmpeg_writer:
    def __init__(self,rtmpUrl,fps,width,height):
        # ffmpeg command
        command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(width, height),
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                # '-preset', 'ultrafast',
                '-f', 'flv', 
                '-flvflags', 'no_duration_filesize',
                rtmpUrl]
        self.pipe=sp.Popen(command, stdin=sp.PIPE)

    def write(self,frame):
        self.pipe.stdin.write(frame.tostring())

    def release(self):
        self.pipe.terminate()

if __name__ == '__main__':
    rtmpUrl = "rtmp://36.152.9.59:39935/live/0?secret=1443ffb0-2f5b-4c85-ad33-55ccfe67c2f1"
    camera_path = "rtsp://192.168.0.69:554/user=admin&password=&channel=1&stream=0.sdp?"
    cap = cv2.VideoCapture(camera_path)

    # Get video information
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 管道配置
    # writer=ffmpeg_writer(rtmpUrl,fps,width,height)
    writer=ffmpeg_writer(rtmpUrl,fps,1024,768)

            
    # read webcamera
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        frame=cv2.resize(frame,(1024,768))
        if not ret:
            print("Opening camera is failed")
            break
                
        # process frame
        # your code
        # process frame
    
        # write to pipe
        writer.write(frame)
    
    writer.release()
