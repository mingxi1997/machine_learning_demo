import math

import cv2
import numpy as np


class ImgChangeAngle:
    def __init__(self, pic, theta, padding=0):
        self.img1 = pic
        self.x = theta
        self.padding = padding

    def left(self, a, b, angle, l1):
        xa = a[0]
        ya = a[1]
        xb = b[0]
        ya1 = self.img1.shape[0] - a[1]
        return [[xa + self.padding, ya1 + self.padding], [(l1 + self.img1.shape[1]) + self.padding, (ya1 - (
                math.tan(math.radians(angle - math.degrees(math.atan((self.img1.shape[0] - ya) / (xb - xa))))) * (
                self.img1.shape[1] - xb + l1))) + self.padding],
                [(l1 + self.img1.shape[1]) + self.padding,
                 ((self.img1.shape[0] - ya) / (xb - xa) * (self.img1.shape[1] - xb + l1) + self.img1.shape[
                     0]) + self.padding],
                [xa + self.padding, ya + self.padding]]

    def right(self, a, b, angle, l1):
        xa = a[0]
        ya = a[1]
        xb = b[0]
        return [[-l1 + self.padding, self.img1.shape[0] - ya - (
                math.tan(math.radians(angle - math.degrees(math.atan((self.img1.shape[0] - ya) / (xa - xb))))) * (
                xa + l1)) + self.padding],
                [xa + self.padding, self.img1.shape[0] - ya + self.padding],
                [xa + self.padding, ya + self.padding],
                [-l1 + self.padding, ya + self.padding + (self.img1.shape[0] - ya) / (xa - xb) * (xa + l1)]]

    def change(self):
        if self.x < 0:
            self.x = - self.x
            x1 = int(25.35 + 8.4 * self.x)
            y1 = int(471.5 - 1.45 * self.x)
            x3 = int(5.3 * self.x + 290.75)
            l1 = int(20.4 * self.x - 66.1)
            xx = [x1, y1]
            yy = [x3, self.img1.shape[0]]
            ptsA = np.float32(
                [[0 + self.padding, 0 + self.padding], [self.img1.shape[1] + self.padding, 0 + self.padding],
                 [self.img1.shape[1] + self.padding, self.img1.shape[0] + self.padding],
                 [0 + self.padding, self.img1.shape[0] + self.padding]]).reshape(-1, 1, 2)
            ptsB = np.float32([self.left(xx, yy, self.x, l1)]).reshape(-1, 1, 2)
        elif self.x > 0:
            x1 = int(25.35 + 8.4 * self.x)
            x1 = self.img1.shape[1] - x1
            y1 = int(471.5 - 1.45 * self.x)
            x3 = int(5.3 * self.x + 290.75)
            x3 = self.img1.shape[1] - x3
            l1 = int(20.4 * self.x - 66.1)
            xx = [x1, y1]
            yy = [x3, self.img1.shape[0]]
            ptsA = np.float32(
                [[0 + self.padding, 0 + self.padding], [self.img1.shape[1] + self.padding, 0 + self.padding],
                 [self.img1.shape[1] + self.padding, self.img1.shape[0] + self.padding],
                 [0 + self.padding, self.img1.shape[0] + self.padding]]).reshape(-1, 1, 2)
            ptsB = np.float32([self.right(xx, yy, self.x, l1)]).reshape(-1, 1, 2)
        else:
            raise TypeError
        ransacReprojThreshold = 4
        self.img1 = cv2.copyMakeBorder(self.img1, self.padding, self.padding, self.padding, self.padding,
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransacReprojThreshold)
        imgOut = cv2.warpPerspective(self.img1, H, (self.img1.shape[1], self.img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return imgOut


img = cv2.imread('test.jpg')
checkimg = ImgChangeAngle(img, 11)
cv2.imshow('test', checkimg.change())
print(checkimg.change().shape)
cv2.waitKey(0)

