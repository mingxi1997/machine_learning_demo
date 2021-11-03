# from utils import nms, scale_coords,letterbox
import cv2
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, r, (dw, dh)

def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):  # 查看是否越界
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


# 坐标对应到原始图像上，反操作：减去pad，除以最小缩放比例
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):  # 输入尺寸，输入坐标，映射的尺寸
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new,计算缩放比率
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                    img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding ，计算扩充的尺寸
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding，减去x方向上的扩充
    coords[:, [1, 3]] -= pad[1]  # y padding，减去y方向上的扩充
    coords[:, :4] /= gain  # 将box坐标对应到原始图像上
    # clip_coords(coords, img0_shape) #边界检查
    return coords


def nms(prediction, conf_thres=0.3, iou_thres=0.6, agnostic=False):
    #  if prediction.dtype is torch.float16:
    #       prediction = prediction.float()  # to FP32
    xc = prediction[..., 4] > conf_thres  # candidates
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        conf = x[:, 5:].max(1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1).reshape(-1, 1)
        # print(conf)
        # print(j)
        # print(conf.shape)
        # print(j.shape)
        j = j.astype(np.float64)
        x = np.concatenate((box, conf, j), 1).reshape(-1)
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        x = x.reshape(-1, 6)
        # print(x)
        # print(x.shape)
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        scores = scores.reshape(-1)
        # print(boxes,scores)
        # print(boxes.shape, scores.shape)
        keep = cpu_nms(boxes, scores, thresh=0.7)
        i = np.array(keep)
        # i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]

    return output


def cpu_nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = scores[:]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep

frame=cv2.imread('test.jpg')



IMAGE_SIZE = 416
anchor_list = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
stride = [8, 16, 32]
CLASSES = 80
CONF_TH = 0.6
NMS_TH = 0.6
area = IMAGE_SIZE *IMAGE_SIZE


model = onnxruntime.InferenceSession("yolov5s.onnx")

img, ratio, (dw, dh)=letterbox(frame,new_shape=IMAGE_SIZE,auto=False)



img = img.transpose(2, 0, 1).astype(np.float32)
img=img/255.0
img = img.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)


inputs = {model.get_inputs()[0].name: img}
pred = model.run(None, inputs)[0]

anchor = np.array(anchor_list).astype(np.float16).reshape(3, -1, 2)
size = [int(area / stride[0] ** 2), int(area / stride[1] ** 2), int(area / stride[2] ** 2)]
feature = [[int(j / stride[i]) for j in (IMAGE_SIZE,IMAGE_SIZE)] for i in range(3)]
y = []
y.append(pred[:, :size[0] * 3, :])
y.append(pred[:, size[0] * 3:size[0] * 3 + size[1] * 3, :])
y.append(pred[:, size[0] * 3 + size[1] * 3:, :])
grid = []
for k, f in enumerate(feature):
    grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])
z = []
for i in range(3):
    src = y[i]
    xy = src[..., 0:2] * 2. - 0.5
    wh = (src[..., 2:4] * 2) ** 2
    dst_xy = []
    dst_wh = []
    for j in range(3):
        dst_xy.append((xy[:, j * size[i]:(j + 1) * size[i], :] + grid[i]) * stride[i])
        dst_wh.append(wh[:, j * size[i]:(j + 1) * size[i], :] * anchor[i][j])
    src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
    src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
    z.append(src.reshape(1, -1, CLASSES + 5))  # 85

pred = np.concatenate(z, 1)
pred = nms(pred, CONF_TH, NMS_TH)


boxes=pred[0]

for box in boxes:
    if dw==0:
        box[1]=(box[1]-dh)
        box[3]=(box[3]-dh)
    elif dh==0:
        box[0]=(box[0]-dw)
        box[2]=(box[2]-dw)
        
color=(0,255,255)        
for box in boxes:
    cv2.rectangle(frame, (int(box[0]/ratio),int(box[1]/ratio)),  (int(box[2]/ratio),int(box[3]/ratio)), color)
    
plt.imshow(frame)











