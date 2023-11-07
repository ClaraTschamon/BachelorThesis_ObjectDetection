import numpy as np
import cv2

# Dein JSON-Dictionary
chessboard_corners = {'a1': {'corner_a': (207.99931, 169.40065), 'corner_b': (245.73703, 169.18), 'corner_c': (241.84544, 198.08734), 'corner_d': (201.86308, 198.26434)}, 'a2': {'corner_a': (245.73703, 169.18), 'corner_b': (286.03387, 168.94498), 'corner_c': (283.8055, 197.90277), 'corner_d': (241.84544, 198.08734)}, 'a3': {'corner_a': (286.03387, 168.94498), 'corner_b': (324.33524, 168.72179), 'corner_c': (324.1664, 197.72575), 'corner_d': (283.8055, 197.90277)}, 'a4': {'corner_a': (324.33524, 168.72179), 'corner_b': (364.72507, 168.48686), 'corner_c': (366.4174, 197.54126), 'corner_d': (324.1664, 197.72575)}, 'a5': {'corner_a': (364.72507, 168.48686), 'corner_b': (403.89206, 168.2594), 'corner_c': (407.6376, 197.362), 'corner_d': (366.4174, 197.54126)}, 'a6': {'corner_a': (403.89206, 168.2594), 'corner_b': (443.587, 168.02922), 'corner_c': (449.25452, 197.18173), 'corner_d': (407.6376, 197.362)}, 'a7': {'corner_a': (443.587, 168.02922), 'corner_b': (485.35855, 167.7873), 'corner_c': (492.82175, 196.9937), 'corner_d': (449.25452, 197.18173)}, 'a8': {'corner_a': (485.35855, 167.7873), 'corner_b': (526.1763, 167.5513), 'corner_c': (535.4959, 196.81029), 'corner_d': (492.82175, 196.9937)}, 'b1': {'corner_a': (201.86308, 198.26434), 'corner_b': (241.84544, 198.08734), 'corner_c': (238.04088, 226.34782), 'corner_d': (195.93149, 226.16493)}, 'b2': {'corner_a': (241.84544, 198.08734), 'corner_b': (283.8055, 197.90277), 'corner_c': (281.602, 226.53745), 'corner_d': (238.04088, 226.34782)}, 'b3': {'corner_a': (283.8055, 197.90277), 'corner_b': (324.1664, 197.72575), 'corner_c': (323.99756, 226.72229), 'corner_d': (281.602, 226.53745)}, 'b4': {'corner_a': (324.1664, 197.72575), 'corner_b': (366.4174, 197.54126), 'corner_c': (368.12827, 226.91502), 'corner_d': (323.99756, 226.72229)}, 'b5': {'corner_a': (366.4174, 197.54126), 'corner_b': (407.6376, 197.362), 'corner_c': (411.46558, 227.10463), 'corner_d': (368.12827, 226.91502)}, 'b6': {'corner_a': (407.6376, 197.362), 'corner_b': (449.25452, 197.18173), 'corner_c': (455.10883, 227.29593), 'corner_d': (411.46558, 227.10463)}, 'b7': {'corner_a': (449.25452, 197.18173), 'corner_b': (492.82175, 196.9937), 'corner_c': (500.616, 227.49565), 'corner_d': (455.10883, 227.29593)}, 'b8': {'corner_a': (492.82175, 196.9937), 'corner_b': (535.4959, 196.81029), 'corner_c': (545.33246, 227.69228), 'corner_d': (500.616, 227.49565)}, 'c1': {'corner_a': (195.93149, 226.16493), 'corner_b': (238.04088, 226.34782), 'corner_c': (233.5431, 259.75845), 'corner_d': (188.78934, 259.76117)}, 'c2': {'corner_a': (238.04088, 226.34782), 'corner_b': (281.602, 226.53745), 'corner_c': (279.0457, 259.7569), 'corner_d': (233.5431, 259.75845)}, 'c3': {'corner_a': (281.602, 226.53745), 'corner_b': (323.99756, 226.72229), 'corner_c': (323.80545, 259.75586), 'corner_d': (279.0457, 259.7569)}, 'c4': {'corner_a': (323.99756, 226.72229), 'corner_b': (368.12827, 226.91502), 'corner_c': (370.04117, 259.75565), 'corner_d': (323.80545, 259.75586)}, 'c5': {'corner_a': (368.12827, 226.91502), 'corner_b': (411.46558, 227.10463), 'corner_c': (415.6679, 259.7561), 'corner_d': (370.04117, 259.75565)}, 'c6': {'corner_a': (411.46558, 227.10463), 'corner_b': (455.10883, 227.29593), 'corner_c': (461.41983, 259.7574), 'corner_d': (415.6679, 259.7561)}, 'c7': {'corner_a': (455.10883, 227.29593), 'corner_b': (500.616, 227.49565), 'corner_c': (508.86053, 259.75937), 'corner_d': (461.41983, 259.7574)}, 'c8': {'corner_a': (500.616, 227.49565), 'corner_b': (545.33246, 227.69228), 'corner_c': (555.54755, 259.76202), 'corner_d': (508.86053, 259.75937)}, 'd1': {'corner_a': (188.78934, 259.76117), 'corner_b': (233.5431, 259.75845), 'corner_c': (228.90053, 294.2444), 'corner_d': (181.45761, 294.24875)}, 'd2': {'corner_a': (233.5431, 259.75845), 'corner_b': (279.0457, 259.7569), 'corner_c': (276.392, 294.24182), 'corner_d': (228.90053, 294.2444)}, 'd3': {'corner_a': (279.0457, 259.7569), 'corner_b': (323.80545, 259.75586), 'corner_c': (323.6048, 294.24), 'corner_d': (276.392, 294.24182)}, 'd4': {'corner_a': (323.80545, 259.75586), 'corner_b': (370.04117, 259.75565), 'corner_c': (372.04974, 294.2394), 'corner_d': (323.6048, 294.24)}, 'd5': {'corner_a': (370.04117, 259.75565), 'corner_b': (415.6679, 259.7561), 'corner_c': (420.10617, 294.2399), 'corner_d': (372.04974, 294.2394)}, 'd6': {'corner_a': (415.6679, 259.7561), 'corner_b': (461.41983, 259.7574), 'corner_c': (468.12393, 294.2415), 'corner_d': (420.10617, 294.2399)}, 'd7': {'corner_a': (461.41983, 259.7574), 'corner_b': (508.86053, 259.75937), 'corner_c': (517.6727, 294.24423), 'corner_d': (468.12393, 294.2415)}, 'd8': {'corner_a': (508.86053, 259.75937), 'corner_b': (555.54755, 259.76202), 'corner_c': (566.5321, 294.248), 'corner_d': (517.6727, 294.24423)}, 'e1': {'corner_a': (181.45761, 294.24875), 'corner_b': (228.90053, 294.2444), 'corner_c': (223.52776, 334.1545), 'corner_d': (172.97318, 334.1585)}, 'e2': {'corner_a': (228.90053, 294.2444), 'corner_b': (276.392, 294.24182), 'corner_c': (273.32086, 334.15216), 'corner_d': (223.52776, 334.1545)}, 'e3': {'corner_a': (276.392, 294.24182), 'corner_b': (323.6048, 294.24), 'corner_c': (323.3726, 334.1505), 'corner_d': (273.32086, 334.15216)}, 'e4': {'corner_a': (323.6048, 294.24), 'corner_b': (372.04974, 294.2394), 'corner_c': (374.3745, 334.15005), 'corner_d': (323.3726, 334.1505)}, 'e5': {'corner_a': (372.04974, 294.2394), 'corner_b': (420.10617, 294.2399), 'corner_c': (425.24277, 334.15054), 'corner_d': (374.3745, 334.15005)}, 'e6': {'corner_a': (420.10617, 294.2399), 'corner_b': (468.12393, 294.2415), 'corner_c': (475.88293, 334.1521), 'corner_d': (425.24277, 334.15054)}, 'e7': {'corner_a': (468.12393, 294.2415), 'corner_b': (517.6727, 294.24423), 'corner_c': (527.87115, 334.1546), 'corner_d': (475.88293, 334.1521)}, 'e8': {'corner_a': (517.6727, 294.24423), 'corner_b': (566.5321, 294.248), 'corner_c': (579.24445, 334.1581), 'corner_d': (527.87115, 334.1546)}, 'f1': {'corner_a': (172.97318, 334.1585), 'corner_b': (223.52776, 334.1545), 'corner_c': (217.67043, 377.6639), 'corner_d': (163.7728, 377.4361)}, 'f2': {'corner_a': (223.52776, 334.1545), 'corner_b': (273.32086, 334.15216), 'corner_c': (269.95532, 377.88785), 'corner_d': (217.67043, 377.6639)}, 'f3': {'corner_a': (273.32086, 334.15216), 'corner_b': (323.3726, 334.1505), 'corner_c': (323.11673, 378.11703), 'corner_d': (269.95532, 377.88785)}, 'f4': {'corner_a': (323.3726, 334.1505), 'corner_b': (374.3745, 334.15005), 'corner_c': (376.9491, 378.35132), 'corner_d': (323.11673, 378.11703)}, 'f5': {'corner_a': (374.3745, 334.15005), 'corner_b': (425.24277, 334.15054), 'corner_c': (430.96207, 378.58847), 'corner_d': (376.9491, 378.35132)}, 'f6': {'corner_a': (425.24277, 334.15054), 'corner_b': (475.88293, 334.1521), 'corner_c': (484.56805, 378.8259), 'corner_d': (430.96207, 378.58847)}, 'f7': {'corner_a': (475.88293, 334.1521), 'corner_b': (527.87115, 334.1546), 'corner_c': (539.3487, 379.0704), 'corner_d': (484.56805, 378.8259)}, 'f8': {'corner_a': (527.87115, 334.1546), 'corner_b': (579.24445, 334.1581), 'corner_c': (593.62787, 379.3146), 'corner_d': (539.3487, 379.0704)}, 'g1': {'corner_a': (163.7728, 377.4361), 'corner_b': (217.67043, 377.6639), 'corner_c': (211.12285, 426.30078), 'corner_d': (153.44821, 426.002)}, 'g2': {'corner_a': (217.67043, 377.6639), 'corner_b': (269.95532, 377.88785), 'corner_c': (266.20786, 426.58743), 'corner_d': (211.12285, 426.30078)}, 'g3': {'corner_a': (269.95532, 377.88785), 'corner_b': (323.11673, 378.11703), 'corner_c': (322.833, 426.8828), 'corner_d': (266.20786, 426.58743)}, 'g4': {'corner_a': (323.11673, 378.11703), 'corner_b': (376.9491, 378.35132), 'corner_c': (379.79333, 427.18103), 'corner_d': (322.833, 426.8828)}, 'g5': {'corner_a': (376.9491, 378.35132), 'corner_b': (430.96207, 378.58847), 'corner_c': (437.2549, 427.48276), 'corner_d': (379.79333, 427.18103)}, 'g6': {'corner_a': (430.96207, 378.58847), 'corner_b': (484.56805, 378.8259), 'corner_c': (494.08575, 427.78217), 'corner_d': (437.2549, 427.48276)}, 'g7': {'corner_a': (484.56805, 378.8259), 'corner_b': (539.3487, 379.0704), 'corner_c': (551.87427, 428.08743), 'corner_d': (494.08575, 427.78217)}, 'g8': {'corner_a': (539.3487, 379.0704), 'corner_b': (593.62787, 379.3146), 'corner_c': (609.2601, 428.3915), 'corner_d': (551.87427, 428.08743)}, 'h1': {'corner_a': (153.44821, 426.002), 'corner_b': (211.12285, 426.30078), 'corner_c': (203.19543, 485.18735), 'corner_d': (140.92992, 484.88666)}, 'h2': {'corner_a': (211.12285, 426.30078), 'corner_b': (266.20786, 426.58743), 'corner_c': (261.67648, 485.47372), 'corner_d': (203.19543, 485.18735)}, 'h3': {'corner_a': (266.20786, 426.58743), 'corner_b': (322.833, 426.8828), 'corner_c': (322.49036, 485.7739), 'corner_d': (261.67648, 485.47372)}, 'h4': {'corner_a': (322.833, 426.8828), 'corner_b': (379.79333, 427.18103), 'corner_c': (383.22388, 486.07672), 'corner_d': (322.49036, 485.7739)}, 'h5': {'corner_a': (379.79333, 427.18103), 'corner_b': (437.2549, 427.48276), 'corner_c': (444.83618, 486.3871), 'corner_d': (383.22388, 486.07672)}, 'h6': {'corner_a': (437.2549, 427.48276), 'corner_b': (494.08575, 427.78217), 'corner_c': (505.53928, 486.69583), 'corner_d': (444.83618, 486.3871)}, 'h7': {'corner_a': (494.08575, 427.78217), 'corner_b': (551.87427, 428.08743), 'corner_c': (566.9313, 487.0108), 'corner_d': (505.53928, 486.69583)}, 'h8': {'corner_a': (551.87427, 428.08743), 'corner_b': (609.2601, 428.3915), 'corner_c': (628.0325, 487.32715), 'corner_d': (566.9313, 487.0108)}}

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


bounding_boxes = [{"xmin":566.8043823242,"ymin":409.0962219238,"xmax":632.2437133789,"ymax":472.0449523926,"confidence":0.9332429767,"class":5,"name":"Black-rook"},{"xmin":539.6846923828,"ymin":303.7201843262,"xmax":590.9473876953,"ymax":367.679901123,"confidence":0.927167654,"class":0,"name":"Black-bishop"},
                  {"xmin":526.9172973633,"ymin":243.3988037109,"xmax":590.8029174805,"ymax":329.3112182617,"confidence":0.9226703644,"class":1,"name":"Black-king"},{"xmin":491.0516662598,"ymin":322.718536377,"xmax":529.158996582,"ymax":370.9422912598,"confidence":0.9224885702,"class":3,"name":"Black-pawn"}]


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
plt.show()


# Create an empty image
image = np.zeros((800, 800, 3), dtype=np.uint8)  # You can adjust the image size as needed

# Project the 3D bounding box corners into 2D image coordinates
for bbox in bounding_boxes:

    # Convert the 2D projected coordinates to integers
    bb_object_points_3d = np.int32(bb_object_points_3d).reshape(-1, 2)

    # Draw the bounding box on the image
    color = (0, 255, 0)  # Green color
    cv2.polylines(image, [bb_object_points_3d], isClosed=True, color=color, thickness=2)

    # Show the image with bounding boxes
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()