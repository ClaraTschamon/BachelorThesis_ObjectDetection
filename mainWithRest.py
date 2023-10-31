import base64
import io
import json
import re
import tkinter as tk
from functools import partial
import requests

import cairosvg
import chess.svg
import cv2
import numpy as np
from PIL import ImageTk, Image


class ChessApp:
    def __init__(self):

        self.url = 'http://127.0.0.1:8080/'

        # Add padding around the grid
        self.padding = 30
        self.grid = None

        self.start_screen = tk.Tk()
        self.start_screen.geometry("1200x750")
        self.start_screen.title("Start Screen")

        self.start_frame = tk.Frame(self.start_screen)
        self.start_frame.pack()

        self.start_button = tk.Button(self.start_frame, text="Start", command=self.detect_board)
        self.start_button.grid(row=0, column=0, padx=self.padding, pady=self.padding)

        # Load the "retake.png" image and resize it
        self.retake_image = Image.open("./assets/retake.png")
        self.retake_image = self.retake_image.resize((25, 25))  # Set the desired width and height
        self.retake_photo = ImageTk.PhotoImage(self.retake_image)
        self.retake_button = tk.Button(self.start_frame, image=self.retake_photo, command=self.detect_board)
        self.retake_button.grid(row=0, column=1, padx=self.padding, pady=self.padding)
        self.retake_button.grid_remove()  # Hide the retake_button initially

        self.image_label = tk.Label(self.start_frame)
        self.image_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)
        self.image_label.grid_remove()  # Hide the image_label initially

        self.info_label = tk.Label(self.start_frame)
        self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)
        self.info_label.grid_remove()  # Hide the info_label initially

        self.is_user_turn = True

        # Load the check and cross images
        self.check_image = Image.open("./assets/checked.png")
        self.check_image = self.check_image.resize((40, 40))
        self.check_image = ImageTk.PhotoImage(self.check_image)

        self.cross_image = Image.open("./assets/cancel.png")
        self.cross_image = self.cross_image.resize((40, 40))
        self.cross_image = ImageTk.PhotoImage(self.cross_image)

        self.turn_text_label = tk.Label(self.start_frame, font=("Helvetica", 20))
        self.turn_text_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)
        self.turn_text_label.grid_remove()

        self.check_cross_label = tk.Label(self.start_frame)
        self.check_cross_label.grid(row=0, column=1, padx=self.padding, pady=self.padding)
        self.check_cross_label.grid_remove()

        self.board = None

        # Row 2: Text
        self.info_label = tk.Label(self.start_frame, text="", font=("Helvetica", 20))
        self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)


    def __get_chessboard_image(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        for _ in range(5):  # Capture and discard the first 5 frames
            cap.read()

        ret, frame = cap.read()
        cap.release()
        return frame

    def __get_chessboard_image_small(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        for _ in range(5):  # Capture and discard the first 5 frames
            cap.read()

        ret, frame = cap.read()
        cap.release()

        # Get the original height and width of the captured frame
        height, width, _ = frame.shape

        # Calculate the scaling factor to make the width 640 pixels
        scale_factor = 640 / width

        # Resize the captured frame width to 640 pixels and calculate new height
        resized_frame = cv2.resize(frame, (640, int(height * scale_factor)))

        # Create a white frame with 640x640 dimensions
        white_frame = np.ones((640, 640, 3), dtype=np.uint8) * 255

        # Calculate the position to paste the resized frame in the center
        x_offset = (640 - resized_frame.shape[1]) // 2
        y_offset = (640 - resized_frame.shape[0]) // 2

        # Paste the resized frame onto the white frame
        white_frame[y_offset:y_offset + resized_frame.shape[0], x_offset:x_offset + resized_frame.shape[1]] = resized_frame

        return white_frame


    def detect_board(self):
        self.grid = None
        frame = self.__get_chessboard_image_small()

        try:
            # Convert the image to bytes for transmission
            _, image_bytes = cv2.imencode('.jpg', frame)
            response = requests.post(self.url+'/detect-board', files={'image': ('image.jpg', image_bytes.tobytes())})
            print("response recieved")
            if response.status_code == 200:
                data = response.json()
                if 'grid' and 'image' in data:
                    grid = data['grid']
                    grid_str = json.loads(grid)

                    # Got grid as string. Need to convert to dictionary
                    key_value_pairs = re.findall(
                        r"'(\w+)': {'corner_a': \(([^)]+)\), 'corner_b': \(([^)]+)\), 'corner_c': \(([^)]+)\), 'corner_d': \(([^)]+)\)}",
                        grid_str)


                    result_dict = {}
                    for key, a, b, c, d in key_value_pairs:
                        result_dict[key] = {
                            'corner_a': tuple(map(float, a.split(', '))),
                            'corner_b': tuple(map(float, b.split(', '))),
                            'corner_c': tuple(map(float, c.split(', '))),
                            'corner_d': tuple(map(float, d.split(', ')))
                        }

                    self.grid = result_dict


                    # Decode the result image from base64
                    result_image_base64 = data['image']
                    result_image_data = base64.b64decode(result_image_base64)
                    result_image = cv2.imdecode(np.fromstring(result_image_data, np.uint8), cv2.IMREAD_COLOR)

                    # Convert the image to a format compatible with Tkinter
                    image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    self.start_button.config(text="Fertig", command=partial(self.__start_game))
                    self.retake_button.grid(row=0, column=1, padx=self.padding, pady=self.padding)
                    self.info_label.config(
                        text="Schachbrett wurde erkannt. Jetzt bitte die Schachfiguren aufstellen und auf 'Fertig' klicken.")
                    # Make the info_label visible
                    self.info_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)

                    # Update the image_label with the new image and make it visible
                    self.image_label.config(image=image)
                    self.image_label.image = image  # Keep a reference to the image to prevent it from being garbage collected
                    self.image_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)

                else:
                    print("Schachbrett konnte nicht erkannt werden")
                    print(data['message'])  # Output the message from the API
                    self.retake_button.grid_remove()
                    self.image_label.grid_remove()
                    self.info_label.config(text="Schachbrett konnte nicht erkannt werden. Bitte erneut versuchen.")
                    self.info_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)

                    # Wait for user input to retry
                    self.start_button.config(text="Erneut versuchen", command=self.detect_board)
                    self.start_button.grid(row=0, column=0, padx=self.padding, pady=self.padding)

            else:
                print('Failed to call the API:', response.status_code)

        except Exception as e:
            print('Error:', e)


    def __detect_figures(self, image):
        # Remove the content from the start frame
        for widget in self.start_frame.winfo_children():
            widget.grid_remove()

        # Convert the image to bytes
        _, image_encoded = cv2.imencode('.jpg', image)
        image_bytes = image_encoded.tobytes()

        # Set the payload data (image and optional grid)
        files = {'image': ('image.jpg', image_bytes)}
        data = {'grid': json.dumps(self.grid)}  # Replace with your grid data if needed

        response = requests.post(self.url+'/inference', files=files, data=data)
        print(response)
        recognized_pieces = response.json()
        return recognized_pieces

    def __start_game(self):
        image = self.__get_chessboard_image_small()
        recognized_pieces = self.__detect_figures(image)
        self.__play_game(recognized_pieces)


    # Function to toggle the turn and update the turn label
    def __toggle_turn(self):
        if self.is_user_turn:
            self.turn_text_label.config(text="Computer ist am Zug!")
            is_user_turn = False
            self.check_cross_label.config(image=self.cross_image)
        else:
            self.turn_text_label.config(text="Du bist am Zug!")
            is_user_turn = True
            self.check_cross_label.config(image=self.check_image)

        return is_user_turn

    def __stop_game(self):
        self.start_button.config(text="Neustart", command=self.__restart_game)

    def __restart_game(self):
        self.__init__()
        self.run()

    def __update_board_img(self, board):
        # Row 1: Chessboard
        svg = chess.svg.board(self.board, size=450)

        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg)
        png_image = Image.open(io.BytesIO(png_data))

        # Display the PNG image in a Tkinter window
        chessboard_image = ImageTk.PhotoImage(png_image)
        chessboard_label = tk.Label(self.start_frame, image=chessboard_image)
        chessboard_label.grid(row=2, column=0, columnspan=3, padx=self.padding, pady=self.padding)

        # Keep a reference to the chessboard image to prevent garbage collection
        chessboard_label.image = chessboard_image

    def check_if_game_over(self):
        is_check_response = requests.get(self.url + '/is-check', params={'fen': self.board.fen()}).json()
        is_checkmate_response = requests.get(self.url + '/is-checkmate', params={'fen': self.board.fen()}).json()

        if is_check_response:
            self.info_label.config(text="Schach!")
            self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)
        elif is_checkmate_response:
            self.info_label.config(text="Schachmatt!")
            self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)
            self.__stop_game()
            # TODO: spiel stoppen -> Neustart button anzeigen
        else:
            self.info_label.config(text="")
            self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)

    def __call_computer_turn(self):
        self.check_if_game_over()

        self.__toggle_turn()
        image = self.__get_chessboard_image_small()
        recognized_pieces = self.__detect_figures(image)

        response = requests.get(self.url + '/make-move', json={'recognized_pieces': recognized_pieces})
        if response.status_code == 200: #todo: wenn game over dann kommt sowieso 400 zurück
            fen = response.text  # Get the content of the response as text
            self.board = chess.Board(fen=fen)
            self.__update_board_img(self.board)
            self.__play_game(None)
        elif response.status_code == 400:
            self.__update_board_img(self.board)
            self.info_label.config(text="Ungültiger Zug! Computer konnte keinen Zug machen")
            self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)
            self.check_if_game_over() # todo: ausprobieren
            self.__play_game(None)

    def __play_game(self, recognized_pieces):

        if recognized_pieces is not None:
            response = requests.get(self.url+'/get-fen', json={'recognized_pieces': recognized_pieces})

            fen = response.text
            self.board = chess.Board(fen=fen)
            self.__update_board_img(self.board)

        # Row 0: Images and Finish Button
        finish_button = tk.Button(self.start_frame, text="Fertig", width=10, height=2, bg="blue", fg="white")
        finish_button.config(command=self.__call_computer_turn)
        finish_button.grid(row=0, column=0, sticky="w", padx=self.padding,
                           pady=self.padding)  # Align the button to the left (west)

        self.retake_button.config(command=self.__start_game)
        self.retake_button.grid(row=0, column=2, padx=self.padding, pady=self.padding)


        self.turn_text_label.config(text="Du bist am Zug!")
        self.turn_text_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)

        # Update the image_label with the new image and make it visible
        self.check_cross_label.config(image=self.check_image)
        self.check_cross_label.grid(row=0, column=1, padx=self.padding, pady=self.padding)


    def run(self):
        self.start_screen.mainloop()


if __name__ == "__main__":
    app = ChessApp()
    app.run()
