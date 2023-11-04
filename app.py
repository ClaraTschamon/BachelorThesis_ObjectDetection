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

def get_image():
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


class ChessApp:
    def __init__(self):
        self.url = 'http://127.0.0.1:8080/'

        # Add padding around the grid
        self.padding = 30
        self.grid = None

        self.main_screen = tk.Tk()
        self.main_screen.geometry("1200x750")
        self.main_screen.title("Schach")

        self.main_frame = tk.Frame(self.main_screen)
        self.main_frame.pack()

        self.start_button = tk.Button(self.main_frame, text="Start", command=self._detect_board)
        self.start_button.grid(row=0, column=0, padx=self.padding, pady=self.padding)

        # Load the "retake.png" image and resize it
        self.retake_image = Image.open("./assets/retake.png")
        self.retake_image = self.retake_image.resize((25, 25))  # Set the desired width and height
        self.retake_photo = ImageTk.PhotoImage(self.retake_image)
        self.retake_button = tk.Button(self.main_frame, image=self.retake_photo, command=self._detect_board)
        self.retake_button.grid(row=0, column=1, padx=self.padding, pady=self.padding)
        self.retake_button.grid_remove()  # Hide the retake_button initially

        self.image_label = tk.Label(self.main_frame)
        self.image_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)
        self.image_label.grid_remove()  # Hide the image_label initially

        self.info_label = tk.Label(self.main_frame)
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

        self.valid_label = tk.Label(self.main_frame, font=("Helvetica", 20))
        self.valid_label.grid(row=1, column=0, padx=self.padding, pady=self.padding)
        self.valid_label.grid_remove()

        self.game_info_label = tk.Label(self.main_frame, font=("Helvetica", 20))
        self.game_info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)
        self.game_info_label.grid_remove()

        self.check_cross_label = tk.Label(self.main_frame)
        self.check_cross_label.grid(row=0, column=1, padx=self.padding, pady=self.padding)
        self.check_cross_label.grid_remove()

        self.board = None

        # Row 2: Text
        self.info_label = tk.Label(self.main_frame, text="", font=("Helvetica", 20))
        self.info_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)

        # spinner for skill level
        self.skill_level = 20
        self.spinner_label = tk.Label(self.main_frame, text="Schwierigkeitsgrad auswählen:")
        self.spinner_label.grid(row=3, column=0, padx=self.padding, pady=self.padding)
        self.spinner_label.grid_remove()

        current_value = tk.StringVar(value=self.skill_level)
        self.skill_level_spinner = tk.Spinbox(self.main_frame, from_=1, to=20, width=5, textvariable=current_value, command=self._update_skill_level)
        self.skill_level_spinner.grid(row=3, column=1, padx=self.padding, pady=self.padding)
        self.skill_level_spinner.grid_remove()

    def _detect_board(self):
        self.grid = None
        frame = get_image()

        # Convert the image to bytes for transmission
        _, image_bytes = cv2.imencode('.jpg', frame)
        response = requests.post(self.url + '/detect-board', files={'image': ('image.jpg', image_bytes.tobytes())})
        if response.status_code == 200:
            data = response.json()
            if 'grid' in data:
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

                self.start_button.config(text="Fertig", command=partial(self._start_game))
                self.retake_button.grid(row=0, column=1, padx=self.padding, pady=self.padding)
                self.info_label.config(
                    text="Schachbrett wurde erkannt. Jetzt bitte die Schachfiguren aufstellen und auf 'Fertig' klicken.")
                # Make the info_label visible
                self.info_label.grid(row=1, column=0, columnspan=2, padx=self.padding, pady=self.padding)

            else:
                self.retake_button.grid_remove()
                self.image_label.grid_remove()
                self.info_label.config(text="Schachbrett konnte nicht erkannt werden. Bitte erneut versuchen.")
                self.info_label.grid(row=1, column=0, columnspan=2, padx=self.padding, pady=self.padding)

                # Wait for user input to retry
                self.start_button.config(text="Erneut versuchen", command=self._detect_board)
                self.start_button.grid(row=0, column=0, padx=self.padding, pady=self.padding)

            # Decode the result image from base64
            result_image_base64 = data['image']
            result_image_data = base64.b64decode(result_image_base64)
            result_image = cv2.imdecode(np.fromstring(result_image_data, np.uint8), cv2.IMREAD_COLOR)

            # Convert the image to a format compatible with Tkinter
            image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            # Update the image_label with the new image and make it visible
            self.image_label.config(image=image)
            self.image_label.image = image  # Keep a reference to the image to prevent it from being garbage collected
            self.image_label.grid(row=2, column=0, padx=self.padding, pady=self.padding)

        else:
            print('Failed to call the API:', response.status_code)

    def _detect_figures(self, image):
        # Remove the content from the start frame
        for widget in self.main_frame.winfo_children():
            widget.grid_remove()

        # Convert the image to bytes
        _, image_encoded = cv2.imencode('.jpg', image)
        image_bytes = image_encoded.tobytes()

        # Set the payload data (image and optional grid)
        files = {'image': ('image.jpg', image_bytes)}
        data = {'grid': json.dumps(self.grid)}  # Replace with your grid data if needed

        response = requests.post(self.url + '/inference', files=files, data=data)
        recognized_pieces = response.json()
        return recognized_pieces

    def _start_game(self):
        image = get_image()
        recognized_pieces = self._detect_figures(image)
        self.info_label.config(text="")
        self._play_game(recognized_pieces)

    # Function to toggle the turn and update the turn label
    def _toggle_turn(self):
        if self.is_user_turn:
            self.valid_label.config(text="Computer ist am Zug!")
            is_user_turn = False
            self.check_cross_label.config(image=self.cross_image)
        else:
            self.valid_label.config(text="Du bist am Zug!")
            is_user_turn = True
            self.check_cross_label.config(image=self.check_image)

        return is_user_turn

    def _stop_game(self):
        self.start_button.config(text="Neustart", command=self._restart_game)

    def _restart_game(self):
        self.__init__()
        self.run()

    def _update_board_img(self):
        # Row 1: Chessboard
        svg = chess.svg.board(self.board, size=450)

        # Convert SVG to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg)
        png_image = Image.open(io.BytesIO(png_data))

        # Display the PNG image in a Tkinter window
        chessboard_image = ImageTk.PhotoImage(png_image)
        chessboard_label = tk.Label(self.main_frame, image=chessboard_image)
        chessboard_label.grid(row=3, column=0, columnspan=3, padx=self.padding, pady=self.padding)

        # Keep a reference to the chessboard image to prevent garbage collection
        chessboard_label.image = chessboard_image

    def _call_computer_turn(self):
        self.info_label.config(text="")

        self._toggle_turn()
        image = get_image()
        recognized_pieces = self._detect_figures(image)

        response = requests.post(self.url + '/make-move', json={'recognized_pieces': recognized_pieces})
        data = response.json()
        fen = data['fen']
        valid = data['valid']
        self.board = chess.Board(fen=fen)
        self.valid_board = valid
        self._update_board_img()

        if valid:
            if 'checkmate' in data:
                checkmate = data['checkmate']
                if checkmate:
                    if checkmate == 'black':
                        self.info_label.config(text="Schwarzer Spieler ist Schachmatt!")
                        self._stop_game()
                    elif checkmate == 'white':
                        self.info_label.config(text="Weißer Spieler ist Schachmatt!")
                        self._stop_game()
                elif 'check' in data:
                    check = data['check']
                    if check:
                        if check == 'black':
                            self.info_label.config(text="Schwarzer Spieler ist Schach!")
                        elif check == 'white':
                            self.info_label.config(text="Weißer Spieler ist Schach!")
                    else:
                        self.info_label.config(text="Dies war ein gültiger Zug.")
            else:
                self.info_label.config(text="Dies war ein gültiger Zug.")

        else:
            self.info_label.config(text="Ungültiger Zug oder ungültige Aufstellung!")

        self._play_game(None)

    def _play_game(self, recognized_pieces):

        if recognized_pieces is not None:
            response = requests.get(self.url + '/get-fen', json={'recognized_pieces': recognized_pieces})
            data = response.json()
            fen = data['fen']
            self.board = chess.Board(fen=fen)
            self.valid_board = data['valid']
            self._update_board_img()

        # Row 0: Images and Finish Button
        finish_button = tk.Button(self.main_frame, text="Fertig", width=10, height=2, bg="blue", fg="white")
        finish_button.config(command=self._call_computer_turn)
        finish_button.grid(row=0, column=0, sticky="w", padx=self.padding,
                           pady=self.padding)  # Align the button to the left (west)

        self.retake_button.config(command=self._start_game)
        self.retake_button.grid(row=0, column=2, padx=self.padding, pady=self.padding)

        if self.valid_board:
            self.valid_label.config(text="Gültiges Brett")
        else:
            self.valid_label.config(text="Ungültiges Brett")
        self.valid_label.grid(row=1, column=0, columnspan=2, pady=self.padding / 2)

        # Update the image_label with the new image and make it visible
        self.check_cross_label.config(image=self.check_image)
        self.check_cross_label.grid(row=0, column=1, padx=self.padding, pady=self.padding)

        self.info_label.grid(row=2, column=0, columnspan=2, pady=self.padding/2)

        self.spinner_label.grid(row=1, column=3, padx=self.padding, pady=self.padding)
        self.skill_level_spinner.grid(row=2, column=3, pady=self.padding/2)

    def _update_skill_level(self):
        new_skill_level = int(self.skill_level_spinner.get())
        if new_skill_level != self.skill_level:
            self.skill_level = new_skill_level
            try:
                response = requests.post(self.url + '/set-skill-level', json={'skill_level': self.skill_level})
                if response.status_code == 200:
                    print(f"Skill level set to {self.skill_level}")
                else:
                    print(f"Failed to set skill level. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to connect to the server: {e}")


    def run(self):
        self.main_screen.mainloop()


if __name__ == "__main__":
    app = ChessApp()
    app.run()
