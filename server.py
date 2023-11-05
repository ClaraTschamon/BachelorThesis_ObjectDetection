import base64

from flask import Flask, request, jsonify
import numpy as np

import chessboardDetection
import chessGame
from chessGame import ChessGame
from inference import *

app = Flask(__name__)

chessboard_detector = chessboardDetection.ChessboardDetection()
chess_game = ChessGame()


@app.route('/detect-board', methods=['POST'])
def detect_chessboard():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    grid, result_image = chessboard_detector.detect_chessboard(image)

    # Encode the result image as a base64 string
    _, result_image_data = cv2.imencode('.png', result_image)
    result_image_base64 = base64.b64encode(result_image_data).decode('utf-8')

    response_data = {'image': result_image_base64}

    if grid is not None:
        response_data['message'] = 'Chessboard detected'
        response_data['grid'] = json.dumps(str(grid))
    else:
        response_data['message'] = 'No chessboard detected'

    return jsonify(response_data)


@app.route('/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        print('No image or grid provided')
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    grid_json = request.form.get('grid')  # Retrieve the grid data as a string
    grid_dict = json.loads(grid_json)  # Parse the grid data as a dictionary

    recognized_pieces = inference_yolov5(image, grid_dict)
    return recognized_pieces


@app.route('/get-fen', methods=['GET'])
def get_fen():
    # Get the recognized pieces from the request
    data = request.get_json()
    recognized_pieces = data.get('recognized_pieces', {})
    fen, valid = chess_game.get_fen_string(recognized_pieces)
    return jsonify({'fen': fen, 'valid': valid})


@app.route('/is-check', methods=['GET'])  # noch nicht getestet
def is_check():
    fen = request.args.get('fen')

    if not fen:
        return jsonify({'error': 'FEN not provided'}), 400

    is_check_result = chessGame.get_is_check(fen)

    return jsonify({'is_check': is_check_result})


@app.route('/is-checkmate', methods=['GET'])  # noch nicht getestet
def is_checkmate():
    fen = request.args.get('fen')

    if not fen:
        return jsonify({'error': 'FEN not provided'}), 400

    is_checkmate_result = chessGame.get_is_checkmate(fen)

    return jsonify({'is_checkmate': is_checkmate_result})


@app.route('/make-move', methods=['POST'])
def make_move():
    try:
        data = request.get_json()
        recognized_pieces = data['recognized_pieces']
        fen, valid, black_check, black_checkmate, white_check, white_checkmate = chess_game.make_move(recognized_pieces)
        response_data = {
            'fen': fen,
            'valid': valid
        }

        if black_checkmate:
            response_data['checkmate'] = 'black'
        elif white_checkmate:
            response_data['checkmate'] = 'white'

        if black_check:
            response_data['check'] = 'black'
        elif white_check:
            response_data['check'] = 'white'

        return jsonify(response_data), 200


    except Exception as e:
        print(f'Exception occurred while making the move: {str(e)}')
        return jsonify({'error': 'Invalid move or error: {}'.format(str(e))}), 400


@app.route('/set-skill-level', methods=['POST'])
def set_skill_level():
    try:
        data = request.get_json()
        skill_level = data['skill_level']
        chess_game.set_skill_level(skill_level)
        return jsonify({'message': 'Skill level updated successfully'})
    except Exception as e:
        return jsonify({'error': 'Invalid skill level or error: {}'.format(str(e))}), 400


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
