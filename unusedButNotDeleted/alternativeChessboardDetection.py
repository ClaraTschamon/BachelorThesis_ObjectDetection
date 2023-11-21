from chessboardDetector import *


class OtherChessboardDetection:

    def __init__(self):
        self.detector = ChessboardDetection()
        pass

    # https://www.freedomvc.com/index.php/2022/01/17/basic-background-remover-with-opencv/
    def _bgremove(self, myimage):
        # First Convert to Grayscale
        myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)

        ret, baseline = cv2.threshold(myimage_grey, 127, 255, cv2.THRESH_TRUNC)

        ret, background = cv2.threshold(baseline, 126, 255, cv2.THRESH_BINARY)

        ret, foreground = cv2.threshold(baseline, 126, 255, cv2.THRESH_BINARY_INV)

        foreground = cv2.bitwise_and(myimage, myimage,
                                     mask=foreground)  # Update foreground with bitwise_and to extract real foreground

        # Convert black and white back into 3 channel greyscale
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # Combine the background and foreground to obtain our final image
        finalimage = background + foreground
        return finalimage

    def _draw_horizontal_vertical_lines(self, lines, image):
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            rho, theta = line[0]
            angle_degrees = np.degrees(theta) + 90  # Convert theta from radians to degrees
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            if -10 <= angle_degrees <= 10 or 175 <= angle_degrees <= 185:  # Horizontal lines
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Color horizontal lines in blue
                horizontal_lines.append((rho, theta))

            elif 90 <= angle_degrees <= 110 or 240 <= angle_degrees <= 270:  # Vertical lines
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Color vertical lines in green
                vertical_lines.append((rho, theta))

        return horizontal_lines, vertical_lines

    def _other_method(self, image):
        image = self._bgremove(image)

        imgGry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, thrash = cv2.threshold(imgGry, 240, 255, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Draw the largest square contour on the image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # draw countour in green

        # Display the result
        display_image('Frame Detection', image)

        # do a hough line detection on only the red lines
        # Define the color range you want to detect (in BGR format)
        lower_color = np.array([0, 255, 0])
        upper_color = np.array([0, 255, 0])

        # Create a binary mask for the desired color
        color_mask = cv2.inRange(image, lower_color, upper_color)

        # Apply Canny edge detection
        edges = cv2.Canny(color_mask, threshold1=60, threshold2=120)

        # Perform Hough Line Detection
        lines = cv2.HoughLines(edges, 0.6, np.pi / 180, threshold=85)

        horizontal_lines, vertical_lines = self._draw_horizontal_vertical_lines(lines, image)

        cv2.imshow('Largest Square Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # detect line intersections
        # https://github.com/andrewleeunderwood/project_MYM/blob/master/cv_chess_functions.py
        points = detect_line_intersections(horizontal_lines, vertical_lines, image)

        draw_points(image, points)

        # create grid and label the cells with chess field names
        grid = define_chess_grid(image, points)

        display_image("chess grid", image)

        return grid, image


if __name__ == "__main__":
    detector = OtherChessboardDetection()
    image = cv2.imread('../testimages/board4.jpg')
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    grid, image = detector._other_method(image)
