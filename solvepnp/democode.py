#!/usr/bin/env python

# Show how to use solvePnP
#  - run camcalib.py first to get camera calibration data

import numpy as np
import cv2


# load previously saved calibration data
calib_data = np.load('./cameraCalibration/CameraParams.npz')
cmx = calib_data['cameraMatrix']
dist = calib_data['dist']

def draw(img, corners, imgpts):

    corner = tuple(corners[0].ravel())
    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 10)
    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 10)
    img = cv2.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 10)

    return img


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# img points
img = cv2.imread('./images/img5.png')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(grey, (9,6), None)

# 3d points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# compute transform
#   - solvePnP requires camera calibraiton
#   - the same info is also returned by calibrateCamera
ret, rvec, tvec = cv2.solvePnP(objp, corners, cmx, dist)

# transform axis to images plane
axis_img,_ = cv2.projectPoints(axis, rvec, tvec, cmx, dist)

# draw axis
draw(img, corners, axis_img)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()