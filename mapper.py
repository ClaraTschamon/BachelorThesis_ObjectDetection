import json



def point_in_square(point, square):
    x, y = point['x_point'], point['y_point']
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
            if point_in_square(point, square_corners):
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

    return recognized_pieces