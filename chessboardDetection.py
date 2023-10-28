from collections import defaultdict

import cv2
import numpy as np
import scipy.cluster as cluster
from scipy import spatial

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
    This function takes an image and finds the largest square-shaped object within it using the OpenCV library.
    - Image Preparation:
        Convert the image to grayscale and blur it slightly to reduce noise.
    - Thresholding:
        Turn the grayscale image into a black and white image where objects are separated from the background.
    - Contour Detection:
        Identify the outlines of objects in the black and white image.
    - Find the Largest Square:
        Among all the detected shapes, pick the one that is the largest and has four corners (a square).
"""


def find_largest_square_contour(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_img, (7, 7), 0)
    adaptive_threshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11,
                                               3)
    contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest square contour
    largest_square_contour = None
    largest_square_area = 0

    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > largest_square_area:
                largest_square_area = area
                largest_square_contour = contour

    return largest_square_contour


"""
   code from: https://github.com/andrewleeunderwood/project_MYM/blob/master/cv_chess_functions.py
   The cluster_points method takes a list of points and groups them into clusters based on their proximity.
   1. Pairwise Distances: It calculates the distances between all pairs of input points.
   2. Hierarchical Clustering: It groups points into clusters using single linkage hierarchical clustering based on distance.
   3. Cluster Assignment: Points closer than a specified threshold are put into the same cluster.
   4. Centroids: For each cluster, it computes the centroid (average) of its points.
   5. Sorting: The cluster centroids are sorted based on their y-coordinates and then their x-coordinates.
"""


def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    #flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance') #for bigger image as input
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 9, 'distance') #for 640x640 image as input
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


"""
    - If horizontal and vertical lines are provided separately, it computes intersections between them by solving equations based on line parameters.
    - If a list of lines is given, it iterates through pairs, avoiding nearly parallel lines, and calculates their intersections.
"""


def detect_line_intersections(horizontal_lines, vertical_lines, lines):
    intersections = []
    if lines is None:
        # https://github.com/andrewleeunderwood/project_MYM/blob/master/cv_chess_functions.py
        for r_h, t_h in horizontal_lines:
            for r_v, t_v in vertical_lines:
                a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
                b = np.array([r_h, r_v])
                inter_point = np.linalg.solve(a, b)
                intersections.append(inter_point)

    else:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]

                rho1, theta1 = line1
                rho2, theta2 = line2

                # Check if lines are nearly parallel (threshold may need adjustment)
                if abs(theta1 - theta2) < np.radians(10):
                    continue

                # Calculate intersection point
                a = np.array([
                    [np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]
                ])
                b = np.array([[rho1], [rho2]])

                try:
                    intersection = np.linalg.solve(a, b)

                    x, y = intersection.ravel()
                    x = int(x)
                    y = int(y)

                    intersections.append((x, y))
                except np.linalg.LinAlgError:
                    # Singular matrix, lines are parallel, continue to the next pair
                    continue

    points = cluster_points(intersections)

    return points


"""
   This method draws the edge points of the chessboard frame and returns their coordinates.
   - It separates points into "top" and "bottom" based on their vertical positions in the image.
   - Sorts these points.
   - Extracts the top two and bottom two points.
   - Draws circles and labels for these four points.
   - Calculates margin points slightly offset from the original points.
   - Returns the coordinates of all these points.
"""


def draw_edgepoints(image, points):
    top_left_margin = None
    top_right_margin = None
    bottom_left_margin = None
    bottom_right_margin = None

    top_points = []
    bottom_points = []

    image_width = image.shape[1]
    image_height = image.shape[0]

    # Filter out points with y-values less than or equal to zero
    filtered_points = [point for point in points if point[1] > 0]

    # Iterate through the filtered list of points
    for point in filtered_points:
        x, y = point

        # Determine if the point is in the top or bottom half of the image
        if y < image_height / 2:
            top_points.append((x, y))
        else:
            bottom_points.append((x, y))

    # Sort the top points based on their y-values (ascending order)
    top_points.sort(key=lambda p: p[1])

    # Sort the bottom points based on their y-values (descending order)
    bottom_points.sort(key=lambda p: p[1], reverse=True)

    # Extract the top two and bottom two points
    top_two_points = top_points[:2]
    bottom_two_points = bottom_points[:2]

    # draw the edge points with a margin of 10px
    margin = 10
    for point in top_two_points + bottom_two_points:
        x, y = point

        if x < image_width / 2 and y < image_height / 2:
            top_left_margin = (x - margin, y - margin)
            cv2.circle(image, (int(top_left_margin[0]), int(top_left_margin[1])), 5, COLOR_GREEN, -1)

        elif x >= image_width / 2 and y < image_height / 2:

            top_right_margin = (x + margin, y - margin)
            cv2.circle(image, (int(top_right_margin[0]), int(top_right_margin[1])), 5, COLOR_GREEN, -1)
        elif x < image_width / 2 and y >= image_height / 2:
            bottom_left_margin = (x - margin, y + margin)
            cv2.circle(image, (int(bottom_left_margin[0]), int(bottom_left_margin[1])), 5, COLOR_GREEN, -1)
        else:
            bottom_right_margin = (x + margin, y + margin)
            cv2.circle(image, (int(bottom_right_margin[0]), int(bottom_right_margin[1])), 5, COLOR_GREEN, -1)

        # circle the edgepoints without the margin
        cv2.circle(image, (int(x), int(y)), 5, COLOR_YELLOW, -1)  # Draw yellow circles

    return top_left_margin, top_right_margin, bottom_left_margin, bottom_right_margin


"""
    Function that draws the given points on the image.
"""


def draw_points(image, points):
    for point in points:
        x, y = point
        cv2.circle(image, (int(x), int(y)), 5, COLOR_YELLOW, -1)  # Draw yellow circles


"""
      -   The function begins by defining the region of interest (ROI) as a polygon with specific vertices. 
          In this case, it's a quadrilateral with vertices specified by the input parameters top_left_margin, 
          top_right_margin, bottom_right_margin, and bottom_left_margin.
      -   It creates a binary mask of the same size as the input canny image, initialized with zeros.
      -   The function fills the ROI area in the mask with white pixels (pixel value 255). This step isolates the area of interest in the mask.
      -   Finally, the function performs a bitwise AND operation between the canny image and the mask. 
          This operation retains only the edges and features within the defined ROI, effectively masking out other areas.
      -   The function returns the resulting ROI image, which contains the edge information within the specified region of interest.
"""


def region_of_interest(canny, top_left_margin, top_right_margin, bottom_right_margin, bottom_left_margin):
    # define the region of interest (roi)
    roi = np.array([
        [[top_left_margin[0], top_left_margin[1]]],
        [[top_right_margin[0], top_right_margin[1]]],
        [[bottom_right_margin[0], bottom_right_margin[1]]],
        [[bottom_left_margin[0], bottom_left_margin[1]]]
    ], dtype=np.int32)

    mask = np.zeros_like(canny)

    # Fill the ROI area with white
    cv2.fillPoly(mask, [roi], COLOR_WHITE)

    # Perform bitwise AND operation to extract ROI
    roi_image = cv2.bitwise_and(canny, mask)

    return roi_image


"""
   The _define_chess_grid function organizes detected intersection points into a 9x9 grid, assigns chessboard cell labels, 
   and records the coordinates of each cell's corners
"""


def define_chess_grid(image, points):
    # Create a 9x9 grid
    grid = np.empty((9, 9), dtype=object)

    # Iterate through points and place them in the grid
    for i, point in enumerate(points):
        row = i // 9  # Calculate the row index
        col = i % 9  # Calculate the column index

        grid[row, col] = point

    # Rearrange points in each row from left to right
    # https://numpy.org/doc/stable/reference/generated/numpy.argsort.html
    for i in range(grid.shape[0]):
        row = grid[i]
        indices = np.argsort([point[0] for point in row])
        grid[i] = row[indices]

    # Initialize an empty dictionary to store the coordinates of each field
    field_coordinates = {}

    # Iterate through rows (from top to bottom) and columns (from left to right)
    for row_index in range(len(grid) - 1):
        for col_index in range(len(grid[0]) - 1):
            # Define the name for the field based on row and column indices
            field_name = chr(ord('a') + row_index) + str(col_index + 1)

            # Extract the coordinates of the edges of the current field
            corner_a = grid[row_index][col_index]
            corner_b = grid[row_index][col_index + 1]
            corner_c = grid[row_index + 1][col_index + 1]
            corner_d = grid[row_index + 1][col_index]

            # Store the coordinates in the dictionary with the field name as the key
            field_coordinates[field_name] = {
                "corner_a": corner_a,
                "corner_b": corner_b,
                "corner_c": corner_c,
                "corner_d": corner_d
            }

    print(field_coordinates)

    # Draw lines connecting the points
    for row in range(grid.shape[0] - 1):
        for col in range(grid.shape[1]):
            point1 = grid[row, col]
            point2 = grid[row + 1, col]
            cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), COLOR_YELLOW, 2)

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1] - 1):
            point1 = grid[row, col]
            point2 = grid[row, col + 1]
            cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), COLOR_YELLOW, 2)

    for field_name, coordinates in field_coordinates.items():
        # Calculate the center point of the field
        center_x = int((coordinates['corner_d'][0] + coordinates['corner_b'][0]) / 2)
        center_y = int((coordinates['corner_d'][1] + coordinates['corner_b'][1]) / 2)
        # Put the field in the calculated center of the field
        cv2.putText(image, field_name, (center_x-5, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 2)

    return field_coordinates


class ChessboardDetection:
    def __init__(self):
        self.max_horizontal_angle_deg = None
        self.min_horizontal_angle_deg = None
        self.max_vertical_angle_deg = None
        self.min_vertical_angle_deg = None


    """
        This method draws in the horizontal and vertical lines detected and skips those which don't fit in the specified range of angles.
        It iterates through the lines detected in the image, where each line is represented by its parameters rho (distance from the origin) and theta (angle with the x-axis).
        For each detected line, it calculates the radians as the angle in radians, and then calculates the coordinates x0, y0, x1, and y2 to draw the line.
        If the line falls within the specified range of horizontal angles (min_horizontal_angle_deg to max_horizontal_angle_deg), it is considered a horizontal line.
        If the line falls within the specified range of vertical angles (min_vertical_angle_deg to max_vertical_angle_deg), it is considered a vertical line.
    """

    def _draw_horizontal_vertical_lines(self, lines, image):
        vertical_lines = []
        horizontal_lines = []

        # Iterate through the new lines and draw them if they meet the criteria
        for line in lines:
            rho, theta = line[0]
            radians = theta
            a = np.cos(radians)
            b = np.sin(radians)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if (self.min_horizontal_angle_deg <= np.degrees(radians) <= self.max_horizontal_angle_deg):
                cv2.line(image, (x1, y1), (x2, y2), COLOR_RED, 2)
                horizontal_lines.append((rho, theta))

            elif (self.min_horizontal_angle_deg + 180 <= np.degrees(radians) <= self.max_horizontal_angle_deg + 180):
                cv2.line(image, (x1, y1), (x2, y2), COLOR_RED, 2)
                horizontal_lines.append((rho, theta))

            elif (self.min_vertical_angle_deg <= np.degrees(radians) <= self.max_vertical_angle_deg):
                cv2.line(image, (x1, y1), (x2, y2), COLOR_BLUE, 2)
                vertical_lines.append((rho, theta))

            elif (self.min_vertical_angle_deg + 180 <= np.degrees(radians) <= self.max_vertical_angle_deg + 180):
                cv2.line(image, (x1, y1), (x2, y2), COLOR_BLUE, 2)
                vertical_lines.append((rho, theta))

            else:
                cv2.line(image, (x1, y1), (x2, y2), COLOR_YELLOW, 2)

        display_image('Detected Lines', image)

        return horizontal_lines, vertical_lines

    """
        this function detects the frame of the chessboard and returns the coordinates of the edgepoints of the frame
    """

    def _detect_frame(self, image):

        # Find the largest square contour in the image
        largest_square_contour = find_largest_square_contour(image)

        # Draw the largest square contour on the image
        if largest_square_contour is None:
            print("No frame detected")
            return None, None, None, None

        cv2.drawContours(image, [largest_square_contour], -1, COLOR_GREEN, 3)

        image = cv2.resize(image, (0, 0), fx=1, fy=1)
        display_image('Largest Square Contour', image)

        # Define the color range you want to detect (in BGR format) (I only want to consider the green lines and do
        # a hough line detection on them)
        lower_color = np.array(COLOR_GREEN)
        upper_color = np.array(COLOR_GREEN)

        # Create a binary mask for the desired color
        color_mask = cv2.inRange(image, lower_color, upper_color)

        # Perform Hough Line Detection
        # rho is the distance resolution of the accumulator in pixels
        # theta is the angle resolution of the accumulator in radians
        # setting theta to np.pi/180 converts degrees to radians
        # threshold specifies the minimum number of votes required for a line to be considered a valid detection
        lines = cv2.HoughLines(color_mask, rho=0.6, theta=np.pi / 180, threshold=120)

        if lines is None:
            print("No lines detected")
            return None, None, None, None

        horizontal_lines = []
        vertical_lines = []

        # Define a small angle tolerance (in degrees) for classifying lines as horizontal or vertical
        angle_tolerance = 45

        # Draw the horizontal and vertical lines on the image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            degrees = np.degrees(theta)

            # Check if the line is approximately horizontal (within angle_tolerance degrees)
            if abs(degrees - 90) < angle_tolerance or abs(degrees + 90) < angle_tolerance:
                vertical_lines.append(line)
                cv2.line(image, (x1, y1), (x2, y2), COLOR_RED, 2)  # Color horizontal lines in red
            # Check if the line is approximately vertical (within angle_tolerance degrees)
            elif abs(degrees) < angle_tolerance or abs(degrees - 180) < angle_tolerance:
                horizontal_lines.append(line)
                cv2.line(image, (x1, y1), (x2, y2), COLOR_BLUE, 2)  # Color vertical lines in blue

        # Find maximum and minimum angles for horizontal lines
        max_horizontal_angle = max(np.degrees(line[0][1]) for line in horizontal_lines)
        min_horizontal_angle = min(np.degrees(line[0][1]) for line in horizontal_lines)

        # Find maximum and minimum angles for vertical lines
        max_vertical_angle = max(np.degrees(line[0][1]) for line in vertical_lines)
        min_vertical_angle = min(np.degrees(line[0][1]) for line in vertical_lines)

        # Consider that the angle of the line can be written in two ways (e.g. 0 degrees and 180 degrees)
        if max_horizontal_angle > 45:
            max_horizontal_angle = max_horizontal_angle - 180
        if min_horizontal_angle > 45:
            min_horizontal_angle = min_horizontal_angle - 180
        if max_vertical_angle > 135:
            max_vertical_angle = max_vertical_angle - 180
        if min_vertical_angle > 135:
            min_vertical_angle = min_vertical_angle - 180

        if max_horizontal_angle < min_horizontal_angle:
            temp = max_horizontal_angle
            max_horizontal_angle = min_horizontal_angle
            min_horizontal_angle = temp
        if max_vertical_angle < min_vertical_angle:
            temp = max_vertical_angle
            max_vertical_angle = min_vertical_angle
            min_vertical_angle = temp

        # store for the grid detection later
        self.max_horizontal_angle_deg = max_horizontal_angle + 2
        self.min_horizontal_angle_deg = min_horizontal_angle - 2
        self.max_vertical_angle_deg = max_vertical_angle + 2
        self.min_vertical_angle_deg = min_vertical_angle - 2

        display_image('Detected Lines', image)

        points = detect_line_intersections(None, None, lines)

        #draw_all_interesection_points(image, points)
        display_image('Clustered Intersection Points', image)

        (top_left_margin,
         top_right_margin,
         bottom_left_margin,
         bottom_right_margin) = draw_edgepoints(image, points)

        display_image('Frame Detection', image)

        return top_left_margin, top_right_margin, bottom_left_margin, bottom_right_margin

    """
        This function is called after the _detect_frame function.
        It detects the chessboard grid and returns the grid and the image with the grid drawn on it.
    """

    def _detect_grid(self, top_left_margin, top_right_margin, bottom_left_margin, bottom_right_margin, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # The (7, 7) tuple represents the kernel size for blurring. You can adjust the kernel size for more or less blurring.
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

        # Adjust the threshold values to filter out more or fewer edges
        canny = cv2.Canny(blurred_image, threshold1=30, threshold2=150)

        #display_image('Canny Edge Detection', canny)

        roi_image = region_of_interest(canny, top_left_margin, top_right_margin, bottom_right_margin,
                                       bottom_left_margin)

        # Probabilistic Hough Line Transform:
        # looks for lines in the edge-detected image
        # rho: This parameter represents the resolution of the accumulator in pixels. It specifies the distance resolution of the Hough grid. In this case, it is set to 1 pixel.
        # theta: This parameter specifies the angle resolution of the accumulator in radians. It determines the angular precision at which lines will be detected. In this case, it is set to np.pi / 180, which corresponds to 1 degree.
        # threshold: This is the threshold value that determines the minimum number of votes (or edge pixels) required for a line to be considered as a detected line.
        # minLineLength: This parameter specifies the minimum length of a line (in pixels) to be considered as a valid line segment
        # maxLineGap: This parameter specifies the maximum gap (in pixels) between line segments that can be connected into a single line.
        linesP = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi / 180, threshold=40, lines=None, minLineLength=10,
                                 maxLineGap=20)

        num_lines = linesP.shape[0]

        if num_lines > 0:
            for i in range(num_lines):
                line = linesP[i][0]
                cv2.line(image, (line[0], line[1]), (line[2], line[3]), COLOR_RED, 2, cv2.LINE_AA)
        else:
            print("No lines detected")

        #display_image('Probabilistic Hough Line Transformation', image)

        # do a hough line detection on only the red lines
        # Define the color range you want to detect (in BGR format)
        lower_color = np.array(COLOR_RED)
        upper_color = np.array(COLOR_RED)

        # Create a binary mask for the desired color
        color_mask = cv2.inRange(image, lower_color, upper_color)

        blurred_image = cv2.GaussianBlur(color_mask, (7, 7), 0)

        #display_image('Blurred Image', blurred_image)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, threshold1=60, threshold2=120)

        #display_image('Canny Edge Detection', edges)

        # Perform Hough Line Detection
        #lines = cv2.HoughLines(edges, 0.5, np.pi / 180, threshold=90) #when image is read in big
        lines = cv2.HoughLines(edges, 0.6, np.pi / 180, threshold=70)

        # Draw the horizontal and vertical lines on the image
        horizontal_lines, vertical_lines = self._draw_horizontal_vertical_lines(lines, image)

        # detect line intersections
        points = detect_line_intersections(horizontal_lines, vertical_lines, None)

        if len(points) != 81:
            print("No chessboard detected")
            return None, image

        # Draw intersection points
        draw_points(image, points)

        display_image('Clustered Intersection Points', image)

        # create grid and label the cells with chess field names
        grid = define_chess_grid(image, points)

        display_image("Chess Grid", image)

        return grid, image

    def detect_chessboard(self, image):
        original_image = image.copy()

        top_left_margin, top_right_margin, bottom_left_margin, bottom_right_margin = self._detect_frame(
            image)
        if top_left_margin is None or top_right_margin is None or bottom_left_margin is None or bottom_right_margin is None:
            print("No chessboard detected")
            return None, image
        else:
            grid, image = self._detect_grid(top_left_margin, top_right_margin, bottom_left_margin, bottom_right_margin,
                                            original_image)
            if grid is not None:
                print("Chessboard detected")
                return grid, image


if __name__ == "__main__":
    detector = ChessboardDetection()
    board_img = cv2.imread('testimages/board3.jpg')
    board_img = cv2.resize(board_img, (0, 0), fx=0.4, fy=0.4)
    grid, image = detector.detect_chessboard(board_img)
