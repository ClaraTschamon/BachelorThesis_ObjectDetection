import json

import numpy as np
import cv2
import torch

import app
import chessboardDetector

import os

#image = app.get_image()
#cv2.imwrite("image.png", image)
image = cv2.imread("image.png")
detector = chessboardDetector.ChessboardDetection()
chessboard_corners, img = detector.detect_chessboard(image)

object_points2d = []
object_points3d = []

for key, point in chessboard_corners.items():
    object_points2d.extend([point['corner_a'], point['corner_b'], point['corner_c'], point['corner_d']])

    #Hier wird angenommen, dass das Schachbrett flach auf der Ebene z=0 liegt, und die 3D-Koordinaten haben daher eine feste z-Koordinate von 0.
    object_points3d.extend([(point['corner_a'][0], point['corner_a'][1], 0),
                            (point['corner_b'][0], point['corner_b'][1], 0),
                            (point['corner_c'][0], point['corner_c'][1], 0),
                            (point['corner_d'][0], point['corner_d'][1], 0)])

object_points2d = np.array(object_points2d, dtype=np.float32) # Bildkoordinatensystem
object_points3d = np.array(object_points3d, dtype=np.float32) # Bildkoordinatensystem

# load previously saved calibration data
#calib_data = np.load('./cameraCalibration/CameraParams.npz')
#cmx = calib_data['cameraMatrix']
#dist = calib_data['dist']

###############################################################################
# Assuming no lens distortion
size = image.shape
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
cmx = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

dist = np.zeros((4, 1))
##########################################################################################

# Schätzung der Kamerapose
retval, rvec, tvec = cv2.solvePnP(object_points3d, object_points2d, cmx, dist)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../training_results/runs/train/results_yolov5m/exp5/weights/best.pt')  # local model


results = model(image)
results.show()
res = results.pandas().xyxy[0]

bounding_boxes = res.to_json(orient="records")
bounding_boxes = json.loads(bounding_boxes) # Bildkoordinatensystem

print('bounding boxes:', bounding_boxes)


bounding_boxes_with_points = []

for box in bounding_boxes:
    xmin = box['xmin']
    ymin = box['ymin']
    xmax = box['xmax']
    ymax = box['ymax']

    # Create an 3D array of the 2D points of the bounding box by adding the height information
    points_3d = np.array([[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, (ymax-ymin)], [xmin, ymax, (ymax-ymin)]], dtype=np.float32) # Bildkoordinatensystem
    # draw the points
    for point in points_3d:
        x, y = point[0], point[1]
        cv2.circle(image, (int(x), int(y)), 7, (0, 0, 0), -1)

    # Project the 3D points of the bounding box back to the 2D image plane
    # Im vorliegenden Code werden die Ecken der Bounding Boxes als 3D-Punkte im Bildkoordinatensystem betrachtet.
    # Diese werden dann mithilfe der Funktion cv2.projectPoints unter Berücksichtigung der Kameraparameter in die 2D-Bildebene projiziert.
    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, cmx, dist)
    print('projected points', projected_points)

    # Draw the projected points on the image
    for point in projected_points:
        x, y = point[0]
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Add the projected points to the bounding box dictionary
    box['projected_points'] = projected_points.tolist()

    # Append the modified bounding box to the new list
    bounding_boxes_with_points.append(box)

cv2.imshow("Projected Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def draw_axes_with_projected_points(image, bounding_boxes):
    for box in bounding_boxes:
        projected_points = np.array(box['projected_points'])

        # Calculate the center of the top edge
        top_center_x = (projected_points[0][0][0] + projected_points[1][0][0]) / 2
        top_center_y = (projected_points[0][0][1] + projected_points[1][0][1]) / 2

        # Calculate the center of the bottom edge
        bottom_center_x = (projected_points[2][0][0] + projected_points[3][0][0]) / 2
        bottom_center_y = (projected_points[2][0][1] + projected_points[3][0][1]) / 2

        # Calculate the height of the bounding box as the Z-axis
        z_height = bottom_center_y - top_center_y

        # Define 3D coordinates for the axes
        origin = (bottom_center_x, bottom_center_y)  # Draw axes at the bottom
        # append the origin to the bounding box
        box['origin'] = origin


        x_end = (bottom_center_x + 20, bottom_center_y)  # X-axis endpoint
        y_end = (bottom_center_x, bottom_center_y + 20)  # Y-axis endpoint
        z_end = (bottom_center_x, bottom_center_y - z_height)  # Z-axis endpoint

        # Draw lines for each axis
        cv2.line(image, (int(origin[0]), int(origin[1])), (int(x_end[0]), int(x_end[1])), (255, 0, 0), 2)  # X-axis (blue)
        cv2.line(image, (int(origin[0]), int(origin[1])), (int(y_end[0]), int(y_end[1])), (0, 255, 255), 2)  # Y-axis (yellow)
        cv2.line(image, (int(origin[0]), int(origin[1])), (int(z_end[0]), int(z_end[1])), (0, 0, 255), 2)  # Z-axis (red)


    return image

image_with_axes = draw_axes_with_projected_points(image, bounding_boxes_with_points)

print("bounding boxes with points:", bounding_boxes_with_points)

cv2.imshow("Bounding Boxes with Axes", image_with_axes)
cv2.waitKey(0)
cv2.destroyAllWindows()

####################################

def point_in_square(point, square):
    x, y = point[0], point[1]
    corner_a_x, corner_a_y = square['corner_a']
    corner_b_x, corner_b_y = square['corner_b']
    corner_c_x, corner_c_y = square['corner_c']
    corner_d_x, corner_d_y = square['corner_d']

    if corner_a_x <= x <= corner_b_x and corner_a_y <= y <= corner_d_y:
        return True
    return False


def map_figures_to_squares(chess_figures, grid):
    points_in_squares = {square_name: [] for square_name in grid.keys()}

    for point in chess_figures:
        for square_name, square_corners in grid.items():
            if point_in_square(point['origin'], square_corners):
                existing_points = points_in_squares[square_name]
                if not existing_points:
                    # If no points exist in the square, add the current point
                    points_in_squares[square_name].append(point)
                else:
                    # Check if the current point has a higher confidence score
                    max_confidence = max(existing_points, key=lambda p: p['confidence'])
                    if point['confidence'] > max_confidence['confidence']:
                        # Replace the existing point with the current point
                        points_in_squares[square_name].remove(max_confidence)
                        points_in_squares[square_name].append(point)

    recognized_pieces = []

    for square_name, points in points_in_squares.items():
        if points:
            max_confidence_point = max(points, key=lambda p: p['confidence'])
            class_name = max_confidence_point['name']
            recognized_pieces.append({'square': square_name, 'class_name': class_name})

    return recognized_pieces


recognized_pieces = map_figures_to_squares(bounding_boxes_with_points, chessboard_corners)
print("recognized pieces:", recognized_pieces)