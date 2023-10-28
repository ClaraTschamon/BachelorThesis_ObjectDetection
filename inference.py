import torch
import cv2
import json
from ultralytics import YOLO


class Inference:

    def __init__(self):
        pass

    def __calculate_points_in_bbox(self, data_json, image):
        # Parse the JSON data
        data = json.loads(data_json)

        # Initialize a list to store the resulting coordinates
        chess_figures = []

        # Iterate through each bounding box and calculate the point
        for item in data:
            try: #YOLOv5 format
                xmin = item["xmin"]
                xmax = item["xmax"]
                ymin = item["ymin"]
                ymax = item["ymax"]
            except KeyError: #YOLOv8 format
                xmin = item["box"]["x1"]
                xmax = item["box"]["x2"]
                ymin = item["box"]["y1"]
                ymax = item["box"]["y2"]

            # Calculate the x and y coordinates of the point
            x_point = int((xmin + xmax) / 2)
            y_point = int(ymin + (3 * (ymax - ymin)) / 4)  # 1/4th from the bottom

            # Draw a point or circle on the image
            cv2.circle(image, (x_point, y_point), 5, (0, 0, 255), -1)  # Red circle with a radius of 5 pixels

            # Get class ID and name
            class_id = item["class"]
            class_name = item["name"]
            confidence = item["confidence"]

            # Append the resulting coordinates to the list
            chess_figures.append({
                "x_point": x_point,
                "y_point": y_point,
                "class_id": class_id,
                "class_name": class_name,
                "confidence:": confidence
            })

        return chess_figures

    def __point_in_square(self, point, square):
        x, y = point['x_point'], point['y_point']
        corner_a_x, corner_a_y = square['corner_a']
        corner_b_x, corner_b_y = square['corner_b']
        corner_c_x, corner_c_y = square['corner_c']
        corner_d_x, corner_d_y = square['corner_d']

        if corner_a_x <= x <= corner_b_x and corner_a_y <= y <= corner_d_y:
            return True
        return False

    def __map_figures_to_squares(self, chess_figures, grid):
        points_in_squares = {square_name: [] for square_name in grid.keys()}

        for point in chess_figures:
            for square_name, square_corners in grid.items():
                if self.__point_in_square(point, square_corners):
                    existing_points = points_in_squares[square_name]
                    if not existing_points:
                        # If no points exist in the square, add the current point
                        points_in_squares[square_name].append(point)
                    else:
                        # Check if the current point has a higher confidence score
                        max_confidence = max(existing_points, key=lambda p: p['confidence:'])
                        if point['confidence:'] > max_confidence['confidence:']:
                            # Replace the existing point with the current point
                            points_in_squares[square_name].remove(max_confidence)
                            points_in_squares[square_name].append(point)

        recognized_pieces = []

        for square_name, points in points_in_squares.items():
            if points:
                max_confidence_point = max(points, key=lambda p: p['confidence:'])
                class_name = max_confidence_point['class_name']
                recognized_pieces.append({'square': square_name, 'class_name': class_name})

        for piece in recognized_pieces:
            print(piece)

        return recognized_pieces

    def inference_yolov5(self, image, grid):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='training_results/runs/train/results_yolov5m/exp3/weights/best.pt')


        #model.conf = 0.7  # NMS confidence threshold
        #iou = 0.45  # NMS IoU threshold
        #agnostic = False  # NMS class-agnostic
        #multi_label = False  # NMS multiple labels per box
        #amp = False  # Automatic Mixed Precision (AMP) inference

        # Inference
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image, augment=True)
        res = results.pandas().xyxy[0]

        data_json = res.to_json(orient="records")

        chess_figures = self.__calculate_points_in_bbox(data_json, image)
        recognized_pieces = self.__map_figures_to_squares(chess_figures, grid)
        return recognized_pieces

    def inference_yolov8(self, image, grid):

        model = YOLO(f'training_results/runs/train/results_yolov8m/exp2/weights/best.pt')

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model(image)

        data_json = results[0].tojson()
        chess_figures = self.__calculate_points_in_bbox(data_json, image)
        recognized_pieces = self.__map_figures_to_squares(chess_figures, grid)
        return recognized_pieces

