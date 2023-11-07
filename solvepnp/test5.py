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

print("object points 3d:", object_points3d)

# load previously saved calibration data
calib_data = np.load('calib.npz')
cmx = calib_data['cmx']
dist = calib_data['dist']

# Schätzung der Kamerapose
retval, rvec, tvec = cv2.solvePnP(object_points3d, object_points2d, cmx, dist)
print(retval, rvec, tvec)


model = torch.hub.load('ultralytics/yolov5', 'custom', path='../training_results/runs/train/results_yolov5m/exp5/weights/best.pt')  # local model

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

results = model(image, augment=True)
res = results.pandas().xyxy[0]

bounding_boxes = res.to_json(orient="records")
bounding_boxes = json.loads(bounding_boxes)

print('bounding boxes:', bounding_boxes)

# Konvertiere die Bounding Box-Koordinaten in die Mitte der Bounding Box
bb_object_points_2d = []

for bbox in bounding_boxes:
    x_center = (bbox["xmin"] + bbox["xmax"]) / 2
    y_center = (bbox["ymin"] + bbox["ymax"]) / 2
    bb_object_points_2d.append((x_center, y_center, 0))

# Konvertiere die Liste der 2D-Punkte in ein NumPy-Array
bb_object_points_2d = np.array(bb_object_points_2d, dtype=np.float32)

# Transformiere die Bounding Box-Punkte in den 3D-Raum relativ zum Schachbrett
bb_object_points_3d = []
for point in bb_object_points_2d:
    point = np.array([point], dtype=np.float32)
    # Projektion der Bounding Box-Punkte in den 3D-Raum
    transformed_point, _ = cv2.projectPoints(point, rvec, tvec, cmx, dist)
    bb_object_points_3d.append(transformed_point[0])

print('bb object points 3d:', bb_object_points_3d)
# Jetzt enthält object_points_3d die Positionen der Bounding Boxes im 3D-Raum relativ zum Schachbrett
# Du kannst diese 3D-Koordinaten verwenden, um die genaue Position der erkannten Figuren im Raum zu bestimmen

##################################################################
# Erstelle ein leeres Dictionary, um die Zuordnung der Bounding Boxes zu den Schachbrettfeldern zu speichern
chessboard_mapping = {}

for bbox_index, bbox_point in enumerate(bb_object_points_3d):
    min_distance = float('inf')
    closest_chessboard = None

    # Gehe durch alle Schachbrettfelder und finde das am nächsten gelegene
    for chessboard_key, chessboard_point in chessboard_corners.items():
        # Extrahiere die 3D-Koordinaten aus dem Dictionary
        chessboard_point = (chessboard_point['corner_a'][0], chessboard_point['corner_a'][1], 0)

        distance = np.linalg.norm(bbox_point - chessboard_point[0])
        if distance < min_distance:
            min_distance = distance
            closest_chessboard = chessboard_key

    # Füge die Zuordnung zur Dictionary hinzu
    chessboard_mapping[bbox_index] = closest_chessboard

print('chessboard mapping:', chessboard_mapping)
# Jetzt enthält chessboard_mapping die Zuordnung der Bounding Boxes zu den Schachbrettfeldern
# Du kannst auf die Schachfigur in jedem Feld zugreifen, indem du chessboard_mapping[BboxIndex] verwendest

#######################################
import numpy as np
import cv2

# Erstellen Sie ein leeres Bild
image = np.zeros((800, 800, 3), dtype=np.uint8)  # Hier wird ein 800x800 Bild erstellt


# Zeichnen Sie die 3D-Punkte object_points3d im Bild
for point in object_points3d:
    # Konvertieren Sie die 3D-Punkte in 2D-Bildkoordinaten
    x = point[0]
    y = point[1]

    # Zeichnen Sie Kreise an den 2D-Koordinaten
    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Grün

for point in bb_object_points_2d:
    x = point[0]
    y = point[1]

    cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Blau

# Zeichnen Sie die 3D-Punkte bb_object_points_3d im Bild
for point in bb_object_points_3d:
    # Konvertieren Sie die 3D-Punkte in 2D-Bildkoordinaten
    x = point[0][0]
    y = point[0][1]

    # Zeichnen Sie Kreise an den 2D-Koordinaten
    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Rot

# Anzeigen des Bildes
cv2.imshow("3D Points in Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

##############################################################################################################
# Annahme: object_points3d enthält die 3D-Koordinaten der Schachbrettfelder
# bb_object_points_3d enthält die 3D-Koordinaten der Mittelpunkte der Bounding Boxes

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2[0])


chessboard_positions = list(chessboard_corners.keys())  # Liste der Schachbrettfelder

# Schleife durch die Bounding Boxes und ordne jedem die entsprechende Schachbrettfeldposition zu
bounding_box_positions = []
for bbox_center in bb_object_points_3d:
    min_distance = float('inf')
    nearest_chessboard_position = None

    for chessboard_position in chessboard_positions:
        chessboard_center = object_points3d[chessboard_positions.index(chessboard_position)]
        distance = euclidean_distance(bbox_center, chessboard_center)

        if distance < min_distance:
            min_distance = distance
            nearest_chessboard_position = chessboard_position

    bounding_box_positions.append(nearest_chessboard_position)

print('bounding box positions:', bounding_box_positions)
# bounding_box_positions enthält jetzt die Schachbrettfeldpositionen, auf denen sich die Schachfiguren befinden


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')  # Use the TkAgg backend (or choose an appropriate one for your system)

# Create a Matplotlib figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points on the chessboard (z = 0)
object_points3d = np.array(object_points3d)
ax.scatter(object_points3d[:, 0], object_points3d[:, 1], object_points3d[:, 2], c='g', marker='o', label='Chessboard Points (z = 0)')

# Extract and plot the positions of objects from bounding boxes
for bbox in bounding_boxes:
    xmin = bbox['xmin']
    xmax = bbox['xmax']
    ymin = bbox['ymin']
    ymax = bbox['ymax']

    # Calculate the height for this bounding box
    height = ymax - ymin

    # Calculate the corners of the object based on the bounding box
    bottom_corners_2d = [(xmin, ymin), (xmax, ymin)]
    top_corners_2d = [(xmin, ymax), (xmax, ymax)]

    # Project the 2D corners into 3D using cv2.projectPoints
    bottom_corners_3d = []
    top_corners_3d = []

    for corner_2d in bottom_corners_2d:
        object_point_2d = np.array([corner_2d[0], corner_2d[1]], dtype=np.float32)
        object_point_3d, _ = cv2.projectPoints(
            np.array([[object_point_2d[0], object_point_2d[1], 0]], dtype=np.float32), rvec, tvec, cmx, dist)
        bottom_corners_3d.append(object_point_3d[0].flatten())

    for corner_2d in top_corners_2d:
        object_point_2d = np.array([corner_2d[0], corner_2d[1]], dtype=np.float32)
        object_point_3d, _ = cv2.projectPoints(
            np.array([[object_point_2d[0], object_point_2d[1], height]], dtype=np.float32), rvec, tvec, cmx, dist)
        top_corners_3d.append(object_point_3d[0].flatten())

    bottom_corners_3d = np.array(bottom_corners_3d)
    top_corners_3d = np.array(top_corners_3d)

    print("bottom corners 3d:", bottom_corners_3d)
    print("top corners 3d:", top_corners_3d)

    # Plot the bottom corners
    ax.plot(bottom_corners_3d[[0, 1, 1, 0, 0], 0], bottom_corners_3d[[0, 1, 1, 0, 0], 1],
            0, c='b')

    # Plot the top corners
    ax.plot(top_corners_3d[[0, 1, 1, 0, 0], 0], top_corners_3d[[0, 1, 1, 0, 0], 1], height,
            c='b')

    # Connect the bottom and top corners to represent the bounding box height
    for i in range(2):
        ax.plot([bottom_corners_3d[i, 0], top_corners_3d[i, 0]], [bottom_corners_3d[i, 1], top_corners_3d[i, 1]],
                [0, height], c='b')

    # Calculate the center point of the bounding box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    center_z = height / 2
    # Plot the center point of the bounding box
    ax.scatter(center_x, center_y, center_z, c='r', marker='o', label='Center Point')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()

# Show the 3D scatter plot
plt.savefig('3d_plot.png')
plt.show()