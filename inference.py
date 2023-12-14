import json
import torch
import cv2
from ultralytics import YOLO
import mapper


def calculate_points_in_bbox(data_json, image):
    # Parse the JSON data
    data = json.loads(data_json)

    # Initialize a list to store the resulting coordinates
    chess_figures = []

    # Iterate through each bounding box and calculate the point
    for item in data:
        try:  # YOLOv5 format
            xmin = item["xmin"]
            xmax = item["xmax"]
            ymin = item["ymin"]
            ymax = item["ymax"]
        except KeyError:  # YOLOv8 format
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


def inference_yolov5(image, grid):
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='training_results/results_yolov5m/exp5/weights/best.pt')

    model.conf = 0.7  # NMS confidence threshold

    # Inference
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)
    res = results.pandas().xyxy[0]

    data_json = res.to_json(orient="records")

    chess_figures = calculate_points_in_bbox(data_json, image)
    recognized_pieces = mapper.map_figures_to_squares(chess_figures, grid)

    return recognized_pieces


def inference_yolov8(image, grid):
    model = YOLO(f'training_results/results_yolov8m/exp2/weights/best.pt')

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image)

    data_json = results[0].tojson()
    chess_figures = calculate_points_in_bbox(data_json, image)
    recognized_pieces = mapper.map_figures_to_squares(chess_figures, grid)
    return recognized_pieces
