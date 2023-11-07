import json
import torch
import numpy as np
import cv2

import app
import chessboardDetection

# Dein JSON-Dictionary
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

print("object points 2d:", object_points2d)
print('object points 3d:', object_points3d)
# load previously saved calibration data
calib_data = np.load('calib.npz')
cmx = calib_data['cmx']
dist = calib_data['dist']
# Schätzung der Kamerapose
retval, rvec, tvec = cv2.solvePnP(object_points3d, object_points2d, cmx, dist)
print(retval, rvec, tvec)

#################################################################

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../training_results/runs/train/results_yolov5m/exp5/weights/best.pt')  # local model

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

results = model(image, augment=True)
results.show()
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
print("object points 2d:", bb_object_points_2d)

# Transformiere die Bounding Box-Punkte in den 3D-Raum relativ zum Schachbrett
bb_object_points_3d = []
for point in bb_object_points_2d:
    point = np.array([point], dtype=np.float32)
    # Projektion der Bounding Box-Punkte in den 3D-Raum
    transformed_point, _ = cv2.projectPoints(point, rvec, tvec, cmx, dist)
    bb_object_points_3d.append(transformed_point[0])

print('object points 3d:', bb_object_points_3d)
# Jetzt enthält object_points_3d die Positionen der Bounding Boxes im 3D-Raum relativ zum Schachbrett
# Du kannst diese 3D-Koordinaten verwenden, um die genaue Position der erkannten Figuren im Raum zu bestimmen


##############################################################################################################
# Annahme: object_points3d enthält die 3D-Koordinaten der Schachbrettfelder
# bb_object_points_3d enthält die 3D-Koordinaten der Mittelpunkte der Bounding Boxes

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2[0])

chessboard_squares = {}  # Ein Dictionary zur Speicherung der Zuordnung von Schachfiguren zu Schachfeldern

for figure_point in bb_object_points_3d:
    min_distance = float('inf')  # Initialisiere mit einer großen Zahl
    closest_square = None

    for square_point in object_points3d:
        distance = calculate_distance(figure_point, square_point)
        if distance < min_distance:
            min_distance = distance
            closest_square = square_point

    # Füge die Schachfigur dem entsprechenden Schachfeld hinzu
    if closest_square is not None:
        # Konvertiere closest_square in ein Tupel und verwende es als Schlüssel
        closest_square = tuple(closest_square)
        if closest_square in chessboard_squares:
            chessboard_squares[closest_square].append(figure_point)
        else:
            chessboard_squares[closest_square] = [figure_point]

print('Zuordnung von Schachfiguren zu Schachfeldern:')
for square, figures in chessboard_squares.items():
    if len(figures) > 0:
        # Hier kannst du den Namen der Figur erhalten (du musst dies basierend auf deinen tatsächlichen Figurennamen anpassen)
        figure_name = ' '.join([str(figure[0]) for figure in figures])
        print(f"Schachfigur {figure_name} auf Schachfeld {square}")


# Erstelle eine Kopie des Bildes, um die Zeichnungen darauf zu machen
annotated_image = image.copy()

# Definiere eine Funktion, um ein Rechteck zu zeichnen
def draw_rectangle(image, point1, point2, color, thickness):
    cv2.rectangle(image, point1, point2, color, thickness)

# Iteriere durch die Eckpunkte des Schachbretts und zeichne die Schachfelder
for key, corners in chessboard_corners.items():
    # Extrahiere die 2D-Koordinaten der Eckpunkte
    corner_a, corner_b, corner_c, corner_d = corners['corner_a'], corners['corner_b'], corners['corner_c'], corners['corner_d']

    # Zeichne ein Rechteck um das Schachfeld
    draw_rectangle(annotated_image, (int(corner_a[0]), int(corner_a[1])), (int(corner_c[0]), int(corner_c[1])), (0, 0, 0), 2)

# Zeige das annotierte Bild
cv2.imshow("Annotated Image", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()