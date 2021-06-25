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
            x1 = int(+8.348571428571429 * self.x + 29.400000000000073)
            y1 = int(-1.7542857142857158 * self.x + 475.20000000000005)
            x3 = int(+6.034285714285713 * self.x + 304.40000000000003)
            l1 = int(+17.160000000000007 * self.x - 47.133333333333354)
            xx = [x1, y1]
            yy = [x3, self.img1.shape[0]]
            ptsA = np.float32(
                [[0 + self.padding, 0 + self.padding], [self.img1.shape[1] + self.padding, 0 + self.padding],
                 [self.img1.shape[1] + self.padding, self.img1.shape[0] + self.padding],
                 [0 + self.padding, self.img1.shape[0] + self.padding]]).reshape(-1, 1, 2)
            ptsB = np.float32([self.left(xx, yy, self.x, l1)]).reshape(-1, 1, 2)
        elif self.x > 0:
            x1 = int(-8.4857142857143 * self.x + 618.666666666667)
            y1 = int(-1.1428571428571421 * self.x + 468.0000000000001)
            x3 = int(-4.645714285714286 * self.x + 362.4666666666665)
            l1 = int(+23.588571428571434 * self.x - 65.1333333333333)
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

checkimg = ImgChangeAngle(img, -11,300)
cv2.imshow('test', checkimg.change())
cv2.waitKey(0)

