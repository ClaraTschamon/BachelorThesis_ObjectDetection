import json

import matplotlib
import numpy as np
import cv2
import torch

import app
import chessboardDetection

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

image = app.get_image()
detector = chessboardDetection.ChessboardDetection()
chessboard_corners, img = detector.detect_chessboard(image)

object_points2d = []
for key in chessboard_corners:
    punkt = chessboard_corners[key]
    object_points2d.append(punkt['corner_a'])
    object_points2d.append(punkt['corner_b'])
    object_points2d.append(punkt['corner_c'])
    object_points2d.append(punkt['corner_d'])

# Convert the list of object_points to a NumPy array
object_points2d = np.array(object_points2d, dtype=np.float32)

# Gehe durch alle Schlüssel im Dictionary
for key in chessboard_corners:
    # Hole den Punkt für den aktuellen Schlüssel
    punkt = chessboard_corners[key]

    # Füge den Z-Wert 0 hinzu
    punkt['corner_a'] = (punkt['corner_a'][0], punkt['corner_a'][1], 0)
    punkt['corner_b'] = (punkt['corner_b'][0], punkt['corner_b'][1], 0)
    punkt['corner_c'] = (punkt['corner_c'][0], punkt['corner_c'][1], 0)
    punkt['corner_d'] = (punkt['corner_d'][0], punkt['corner_d'][1], 0)


# Extract and flatten 3D coordinates from the dictionary
object_points3d = []
for key in chessboard_corners:
    punkt = chessboard_corners[key]
    object_points3d.append(punkt['corner_a'])
    object_points3d.append(punkt['corner_b'])
    object_points3d.append(punkt['corner_c'])
    object_points3d.append(punkt['corner_d'])

# Convert the list of object_points to a NumPy array
object_points3d = np.array(object_points3d, dtype=np.float32)

# load previously saved calibration data
calib_data = np.load('calib.npz')
cmx = calib_data['cmx']
dist = calib_data['dist']

# Schätzung der Kamerapose
retval, rvec, tvec = cv2.solvePnP(object_points3d, object_points2d, cmx, dist)

def draw(img, rvec, tvec, cmx, dist):
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    # Transform axis to image plane
    axis_img, _ = cv2.projectPoints(axis, rvec, tvec, cmx, dist)

    # Draw lines for the pose
    img_center = (img.shape[1] // 2, img.shape[0] // 2)
    img = cv2.line(img, img_center, tuple(map(int, axis_img[0].ravel())), (255, 0, 0), 10)
    img = cv2.line(img, img_center, tuple(map(int, axis_img[1].ravel())), (0, 255, 0), 10)
    img = cv2.line(img, img_center, tuple(map(int, axis_img[2].ravel())), (0, 0, 255), 10)

    return img


# Draw the axis on the image
img_with_pose = draw(img, rvec, tvec, cmx, dist)

# Display the image
cv2.imshow('Image with Pose', img_with_pose)
cv2.waitKey()
cv2.destroyAllWindows()
